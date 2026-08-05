from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lol_ai.config import ARTIFACTS_DIR, FILTERED_DATA_FILE, PROCESSED_DATA_DIR
from lol_ai.modeling.features import TEAM_POSITIONS, normalize_text


PLAYER_RATINGS_REPORT = ARTIFACTS_DIR / "reports" / "player_impact_ratings.csv"
PLAYER_HISTORY_FILE = PROCESSED_DATA_DIR / "cblol_game_context_dataset.csv"

PLAYER_METRICS = [
    "result",
    "damageshare",
    "earnedgoldshare",
    "dpm",
    "vspm",
    "cspm",
    "wardsplaced",
    "visionscore",
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
]


@dataclass(frozen=True)
class LineupRating:
    team_name: str
    players: dict[str, str]
    player_ratings: dict[str, dict[str, Any]]
    lineup_rating: float
    baseline_lineup_rating: float
    lineup_delta: float
    missing_players: list[str]


def _resolve_filtered_path(input_path: Path | None = None) -> Path:
    if input_path is not None:
        return input_path
    if FILTERED_DATA_FILE.exists():
        return FILTERED_DATA_FILE
    raise FileNotFoundError(f"Arquivo filtrado não encontrado: {FILTERED_DATA_FILE}")


def _resolve_context_path(input_path: Path | None = None) -> Path:
    if input_path is not None:
        return input_path
    if PLAYER_HISTORY_FILE.exists():
        return PLAYER_HISTORY_FILE
    raise FileNotFoundError(f"Arquivo contextual não encontrado: {PLAYER_HISTORY_FILE}")


def _percentile_by_position(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("position")[column].rank(pct=True, method="average")


def build_player_ratings(
    filtered_path: Path | None = None,
    *,
    frame: pd.DataFrame | None = None,
    write_report: bool = True,
) -> pd.DataFrame:
    if frame is None:
        source_path = _resolve_filtered_path(filtered_path)
        frame = pd.read_csv(source_path)
    frame = frame[frame["position"].isin(TEAM_POSITIONS)].copy()

    if frame.empty:
        raise ValueError("Nenhuma linha de jogador foi encontrada para montar o medidor de impacto.")

    for column in PLAYER_METRICS:
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    frame["early_advantage"] = frame[["golddiffat15", "xpdiffat15", "csdiffat15"]].mean(axis=1)

    percentile_columns = {}
    for column in ["result", "damageshare", "earnedgoldshare", "dpm", "vspm", "cspm", "wardsplaced", "visionscore", "early_advantage"]:
        percentile_columns[column] = _percentile_by_position(frame, column)

    frame["game_impact_score"] = (
        0.30 * percentile_columns["result"]
        + 0.18 * percentile_columns["damageshare"]
        + 0.12 * percentile_columns["earnedgoldshare"]
        + 0.12 * percentile_columns["dpm"]
        + 0.10 * percentile_columns["vspm"]
        + 0.08 * percentile_columns["cspm"]
        + 0.05 * percentile_columns["wardsplaced"]
        + 0.05 * percentile_columns["visionscore"]
        + 0.10 * percentile_columns["early_advantage"]
    ) * 100

    grouping_columns = ["playerid", "playername", "position"]
    if "teamname" in frame.columns:
        grouping_columns.append("teamname")

    aggregated = (
        frame.groupby(grouping_columns, as_index=False)
        .agg(
            games=("gameid", "count"),
            wins=("result", "sum"),
            game_impact_score=("game_impact_score", "mean"),
            damageshare=("damageshare", "mean"),
            earnedgoldshare=("earnedgoldshare", "mean"),
            dpm=("dpm", "mean"),
            vspm=("vspm", "mean"),
            cspm=("cspm", "mean"),
            wardsplaced=("wardsplaced", "mean"),
            visionscore=("visionscore", "mean"),
        )
        .reset_index(drop=True)
    )

    aggregated["winrate"] = aggregated["wins"] / aggregated["games"].clip(lower=1)
    reliability = aggregated["games"] / (aggregated["games"] + 8)
    aggregated["impact_score"] = 50 + (aggregated["game_impact_score"] - 50) * reliability
    aggregated["impact_score"] = aggregated["impact_score"].clip(0, 100).round(2)

    latest_rows = frame.sort_values("date").groupby("playerid", as_index=False).tail(1)
    latest_team_lookup = latest_rows.set_index("playerid")["teamname"].to_dict() if "teamname" in latest_rows.columns else {}
    aggregated["latest_team"] = aggregated["playerid"].map(latest_team_lookup).fillna("")

    if write_report:
        PLAYER_RATINGS_REPORT.parent.mkdir(parents=True, exist_ok=True)
        aggregated.sort_values(["impact_score", "games"], ascending=[False, False]).to_csv(PLAYER_RATINGS_REPORT, index=False)
    return aggregated.sort_values(["impact_score", "games"], ascending=[False, False]).reset_index(drop=True)


def build_impact_lookup(
    filtered_path: Path | None = None,
    cutoff_date: object | None = None,
) -> dict[tuple[str, str], float]:
    source_path = _resolve_filtered_path(filtered_path)
    frame = pd.read_csv(source_path)
    if cutoff_date is not None:
        frame = frame[pd.to_datetime(frame["date"], errors="coerce") < pd.Timestamp(cutoff_date)]
    frame = frame[frame["position"].isin(TEAM_POSITIONS)]
    if frame.empty:
        return {}
    ratings = build_player_ratings(frame=frame.copy(), write_report=False)
    return {
        (str(row["playername"]).strip().lower(), str(row["position"]).strip().lower()): float(row["impact_score"])
        for _, row in ratings.iterrows()
    }


def load_player_ratings(filtered_path: Path | None = None) -> pd.DataFrame:
    if PLAYER_RATINGS_REPORT.exists():
        return pd.read_csv(PLAYER_RATINGS_REPORT)
    return build_player_ratings(filtered_path)


def latest_lineup_for_team(context_frame: pd.DataFrame, team_name: str) -> dict[str, str]:
    team_games = context_frame[(context_frame["blue_team"] == team_name) | (context_frame["red_team"] == team_name)].sort_values("date")
    if team_games.empty:
        return {}

    latest = team_games.iloc[-1]
    if latest["blue_team"] == team_name:
        prefix = "blue"
    else:
        prefix = "red"

    return {position: normalize_text(latest.get(f"{prefix}_{position}_player")) for position in TEAM_POSITIONS}


def lookup_player_rating(ratings_frame: pd.DataFrame, player_name: str, position: str) -> dict[str, Any] | None:
    candidates = ratings_frame[
        (ratings_frame["playername"].astype(str).str.lower() == normalize_text(player_name).lower())
        & (ratings_frame["position"].astype(str).str.lower() == normalize_text(position).lower())
    ]
    if candidates.empty:
        return None
    return candidates.sort_values(["impact_score", "games"], ascending=[False, False]).iloc[0].to_dict()


def evaluate_lineup(
    team_name: str,
    lineup: dict[str, str] | None,
    ratings_frame: pd.DataFrame,
    context_frame: pd.DataFrame,
) -> LineupRating:
    baseline_players = latest_lineup_for_team(context_frame, team_name)
    selected_players = lineup or baseline_players

    player_ratings: dict[str, dict[str, Any]] = {}
    missing_players: list[str] = []
    total_score = 0.0
    baseline_total_score = 0.0
    player_count = 0

    for position in TEAM_POSITIONS:
        player_name = selected_players.get(position, "")
        baseline_name = baseline_players.get(position, "")

        rating_row = lookup_player_rating(ratings_frame, player_name, position) if player_name else None
        baseline_rating_row = lookup_player_rating(ratings_frame, baseline_name, position) if baseline_name else None

        if rating_row is None:
            missing_players.append(player_name or f"{team_name} {position}")
            selected_score = 50.0
        else:
            selected_score = float(rating_row["impact_score"])
            player_ratings[position] = rating_row

        if baseline_rating_row is None:
            baseline_score = 50.0
        else:
            baseline_score = float(baseline_rating_row["impact_score"])

        total_score += selected_score
        baseline_total_score += baseline_score
        player_count += 1

    lineup_rating = round(total_score / max(player_count, 1), 2)
    baseline_lineup_rating = round(baseline_total_score / max(player_count, 1), 2)
    lineup_delta = round(lineup_rating - baseline_lineup_rating, 2)

    return LineupRating(
        team_name=team_name,
        players=selected_players,
        player_ratings=player_ratings,
        lineup_rating=lineup_rating,
        baseline_lineup_rating=baseline_lineup_rating,
        lineup_delta=lineup_delta,
        missing_players=missing_players,
    )
