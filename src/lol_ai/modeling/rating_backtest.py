from __future__ import annotations

import math
from itertools import product

import pandas as pd
from sklearn.metrics import log_loss

from lol_ai.modeling.rating import TEAM_POSITIONS, EloConfig, RatingEngine

K_GRID = (16.0, 24.0, 32.0, 40.0)
SEASON_CARRY_GRID = (0.4, 0.6, 0.8, 1.0)
ROSTER_REGRESSION_GRID = (0.0, 0.05, 0.10, 0.20)
IMPACT_SCALE_GRID = (0.0, 1.0, 2.0)


def estimate_side_advantage(blue_win_rate: float) -> float:
    clamped = min(max(blue_win_rate, 0.05), 0.95)
    return -400.0 * math.log10(1.0 / clamped - 1.0)


def _lineup_from_row(row: pd.Series, prefix: str) -> dict[str, str]:
    return {
        position: str(row.get(f"{prefix}_{position}_player") or "").strip()
        for position in TEAM_POSITIONS
    }


def run_walk_forward(
    frame: pd.DataFrame,
    config: EloConfig,
    impact_lookup: dict[tuple[str, str], float],
) -> tuple[RatingEngine, pd.Series]:
    engine = RatingEngine(config, impact_lookup)
    ordered = frame.sort_values(["date", "series_id", "game_number"])
    probabilities: dict[object, float] = {}
    for index, row in ordered.iterrows():
        probabilities[index] = engine.process_game(
            date=row["date"],
            league=str(row.get("league", "")),
            year=int(row["year"]),
            blue_team=str(row["blue_team"]),
            red_team=str(row["red_team"]),
            blue_lineup=_lineup_from_row(row, "blue"),
            red_lineup=_lineup_from_row(row, "red"),
            blue_win=bool(int(row["blue_win"])),
        )
    return engine, pd.Series(probabilities).reindex(frame.index)


def calibrate_config(
    frame: pd.DataFrame,
    validation_index: pd.Index,
    impact_lookup: dict[tuple[str, str], float],
    side_advantage: float,
) -> EloConfig:
    best_config: EloConfig | None = None
    best_loss = float("inf")
    y_validation = frame.loc[validation_index, "blue_win"].astype(int)
    for k, carry, roster, impact in product(K_GRID, SEASON_CARRY_GRID, ROSTER_REGRESSION_GRID, IMPACT_SCALE_GRID):
        config = EloConfig(
            k=k,
            season_carry=carry,
            roster_regression_per_player=roster,
            impact_scale=impact,
            side_advantage=side_advantage,
        )
        _, probabilities = run_walk_forward(frame, config, impact_lookup)
        loss = float(log_loss(y_validation, probabilities.loc[validation_index], labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_config = config
    assert best_config is not None
    return best_config


def series_level_accuracy(
    frame: pd.DataFrame,
    probabilities: pd.Series,
    evaluation_index: pd.Index,
) -> float | None:
    rows = frame.loc[evaluation_index].copy()
    rows["blue_prob"] = probabilities.loc[evaluation_index]
    hits: list[bool] = []
    for _, group in rows.groupby("series_id"):
        ordered = group.sort_values("game_number")
        first = ordered.iloc[0]
        predicted = first["blue_team"] if first["blue_prob"] >= 0.5 else first["red_team"]
        actual = ordered["winner_team"].value_counts().idxmax()
        hits.append(predicted == actual)
    if not hits:
        return None
    return float(sum(hits) / len(hits))
