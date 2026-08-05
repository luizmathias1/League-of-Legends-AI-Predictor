from __future__ import annotations

import json
from collections import Counter
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lol_ai.config import MODEL_ARTIFACTS_DIR, PROCESSED_DATA_DIR, LEGACY_PROCESSED_FILE, RATING_CONFIG_FILE
from lol_ai.modeling.features import build_feature_frame, load_context_dataset, normalize_text
from lol_ai.modeling.player_impact import build_impact_lookup, build_player_ratings, evaluate_lineup
from lol_ai.modeling.rating import EloConfig, expected_score
from lol_ai.modeling.rating_backtest import blend_probabilities, run_walk_forward
from lol_ai.modeling.series import series_probabilities


@dataclass(frozen=True)
class MatchPrediction:
    blue_team: str
    red_team: str
    base_blue_win_probability: float
    blue_win_probability: float
    red_win_probability: float
    model_probability: float
    matchup_summary: dict[str, Any]
    best_of: int
    series_win_probability_blue: float
    series_win_probability_red: float
    likely_series_score_blue: str
    likely_series_score_red: str
    game_win_probabilities: list[dict[str, float]]
    common_bans: dict[str, list[dict[str, Any]]]
    common_picks: dict[str, list[dict[str, Any]]]
    lineup_summary: dict[str, Any]


def _resolve_dataset_path(input_path: Path | None = None) -> Path:
    if input_path is not None:
        return input_path
    preferred = PROCESSED_DATA_DIR / "cblol_game_context_dataset.csv"
    if preferred.exists():
        return preferred
    return LEGACY_PROCESSED_FILE


def _load_artifacts() -> tuple[Any, list[str], Any]:
    with (MODEL_ARTIFACTS_DIR / "cblol_preprocessor.pkl").open("rb") as handle:
        preprocessor = pickle.load(handle)
    with (MODEL_ARTIFACTS_DIR / "cblol_logistic_regression.pkl").open("rb") as handle:
        logistic_model = pickle.load(handle)
    feature_names = json.loads((MODEL_ARTIFACTS_DIR / "cblol_feature_names.json").read_text(encoding="utf-8"))
    return preprocessor, feature_names, logistic_model


def _team_history_summary(frame: pd.DataFrame, blue_team: str, red_team: str) -> dict[str, Any]:
    blue_history = frame[(frame["blue_team"] == blue_team) | (frame["red_team"] == blue_team)].copy()
    red_history = frame[(frame["blue_team"] == red_team) | (frame["red_team"] == red_team)].copy()

    def last_n_winrate(team_frame: pd.DataFrame, team_name: str, n: int = 10) -> float | None:
        if team_frame.empty:
            return None
        results: list[bool] = []
        ordered = team_frame.sort_values("date")
        for _, row in ordered.tail(n).iterrows():
            if row["blue_team"] == team_name:
                results.append(bool(row["blue_win"]))
            elif row["red_team"] == team_name:
                results.append(not bool(row["blue_win"]))
        if not results:
            return None
        return float(sum(results) / len(results))

    h2h = frame[
        ((frame["blue_team"] == blue_team) & (frame["red_team"] == red_team))
        | ((frame["blue_team"] == red_team) & (frame["red_team"] == blue_team))
    ].sort_values("date")

    h2h_results: list[str] = []
    for _, row in h2h.tail(10).iterrows():
        winner = row["blue_team"] if bool(row["blue_win"]) else row["red_team"]
        h2h_results.append(winner)

    return {
        "blue_last10_winrate": last_n_winrate(blue_history, blue_team, 10),
        "red_last10_winrate": last_n_winrate(red_history, red_team, 10),
        "h2h_last10_games": len(h2h_results),
        "h2h_last10_blue_winrate": None if not h2h_results else float(sum(1 for winner in h2h_results if winner == blue_team) / len(h2h_results)),
        "h2h_last10_red_winrate": None if not h2h_results else float(sum(1 for winner in h2h_results if winner == red_team) / len(h2h_results)),
        "recent_h2h_winners": h2h_results,
    }


def _team_event_counts(frame: pd.DataFrame, team_name: str, event_type: str, n_games: int = 10) -> list[dict[str, Any]]:
    team_history = frame[(frame["blue_team"] == team_name) | (frame["red_team"] == team_name)].copy()
    if team_history.empty:
        return []

    ordered = team_history.sort_values("date").tail(n_games)
    counter: Counter[str] = Counter()
    for _, row in ordered.iterrows():
        if row["blue_team"] == team_name:
            column_name = f"blue_{event_type}"
        else:
            column_name = f"red_{event_type}"

        raw_value = row[column_name]
        if not isinstance(raw_value, str):
            continue
        for item in raw_value.split(";"):
            cleaned = item.strip()
            if cleaned:
                counter[cleaned] += 1

    total = sum(counter.values()) or 1
    return [
        {"name": name, "count": count, "share": round(count / total, 4)}
        for name, count in counter.most_common(6)
    ]


def _best_of_five_probabilities(per_game_probability: float) -> dict[str, float]:
    p = per_game_probability
    q = 1 - p
    blue_3_0 = p**3
    blue_3_1 = 3 * p**3 * q
    blue_3_2 = 6 * p**3 * (q**2)
    red_3_0 = q**3
    red_3_1 = 3 * q**3 * p
    red_3_2 = 6 * q**3 * (p**2)

    return {
        "blue_3_0": blue_3_0,
        "blue_3_1": blue_3_1,
        "blue_3_2": blue_3_2,
        "red_3_0": red_3_0,
        "red_3_1": red_3_1,
        "red_3_2": red_3_2,
        "blue_series_win": blue_3_0 + blue_3_1 + blue_3_2,
        "red_series_win": red_3_0 + red_3_1 + red_3_2,
    }


def _most_likely_scoreline(probabilities: dict[str, float]) -> tuple[str, str, str]:
    candidates = {
        "3x0": probabilities["blue_3_0"],
        "3x1": probabilities["blue_3_1"],
        "3x2": probabilities["blue_3_2"],
        "0x3": probabilities["red_3_0"],
        "1x3": probabilities["red_3_1"],
        "2x3": probabilities["red_3_2"],
    }
    best_score = max(candidates, key=candidates.get)
    reverse_score = {
        "3x0": "0x3",
        "3x1": "1x3",
        "3x2": "2x3",
        "0x3": "3x0",
        "1x3": "3x1",
        "2x3": "3x2",
    }[best_score]
    return best_score, reverse_score, f"{candidates[best_score]:.2%}"


def _build_hypothetical_row(frame: pd.DataFrame, blue_team: str, red_team: str) -> pd.DataFrame:
    sorted_frame = frame.sort_values("date")
    blue_history = sorted_frame[(sorted_frame["blue_team"] == blue_team) | (sorted_frame["red_team"] == blue_team)]
    red_history = sorted_frame[(sorted_frame["blue_team"] == red_team) | (sorted_frame["red_team"] == red_team)]

    def recent_winrate(team_history: pd.DataFrame, team_name: str, n: int = 10) -> float:
        results: list[bool] = []
        for _, row in team_history.tail(n).iterrows():
            if row["blue_team"] == team_name:
                results.append(bool(row["blue_win"]))
            elif row["red_team"] == team_name:
                results.append(not bool(row["blue_win"]))
        return float(sum(results) / len(results)) if results else 0.5

    def role_recent_winrate(team_name: str, position: str, n: int = 10) -> float:
        subset = sorted_frame[
            ((sorted_frame["blue_team"] == team_name) & (sorted_frame["blue_" + position + "_player"].notna()))
            | ((sorted_frame["red_team"] == team_name) & (sorted_frame["red_" + position + "_player"].notna()))
        ]
        if subset.empty:
            return 0.5
        # fallback to team recent form, because role-to-role history is not directly stored in the context table
        return recent_winrate(team_history=team_history_frame(frame, team_name), team_name=team_name, n=n)

    def team_history_frame(full_frame: pd.DataFrame, team_name: str) -> pd.DataFrame:
        return full_frame[(full_frame["blue_team"] == team_name) | (full_frame["red_team"] == team_name)]

    team_history_blue = team_history_frame(sorted_frame, blue_team)
    team_history_red = team_history_frame(sorted_frame, red_team)

    last_blue_date = sorted_frame[sorted_frame["blue_team"].isin([blue_team, red_team]) | sorted_frame["red_team"].isin([blue_team, red_team])]["date"].max()
    if pd.isna(last_blue_date):
        last_blue_date = pd.Timestamp.utcnow()

    row = {
        "series_id": f"PRED|{blue_team} vs {red_team}",
        "game_id": f"PRED|{blue_team}|{red_team}",
        "game_number": 1,
        "series_game_count": 1,
        "best_of": 1,
        "league": "CBLOL",
        "year": int(last_blue_date.year),
        "split": "Future Match",
        "playoffs": 0,
        "date": last_blue_date.strftime("%Y-%m-%d %H:%M:%S"),
        "patch": "",
        "blue_team": blue_team,
        "red_team": red_team,
        "blue_win": 0,
        "winner_team": "",
        "series_games_played_before": 0,
        "series_score_blue_before": 0,
        "series_score_red_before": 0,
        "first_pick_side": "",
        "blue_bans": "",
        "red_bans": "",
        "blue_picks": "",
        "red_picks": "",
        "blue_last5_winrate": recent_winrate(team_history_blue, blue_team, 5),
        "blue_last10_winrate": recent_winrate(team_history_blue, blue_team, 10),
        "red_last5_winrate": recent_winrate(team_history_red, red_team, 5),
        "red_last10_winrate": recent_winrate(team_history_red, red_team, 10),
        "blue_h2h_last10_winrate": 0.5,
        "red_h2h_last10_winrate": 0.5,
        "context_text": f"Future match: {blue_team} vs {red_team}",
    }

    for position in ("top", "jng", "mid", "bot", "sup"):
        row[f"blue_{position}_last10_winrate"] = recent_winrate(team_history_blue, blue_team, 10)
        row[f"red_{position}_last10_winrate"] = recent_winrate(team_history_red, red_team, 10)
        row[f"blue_{position}_player"] = ""
        row[f"red_{position}_player"] = ""

    return pd.DataFrame([row])


def predict_matchup(
    blue_team: str,
    red_team: str,
    blue_lineup: list[str] | None = None,
    red_lineup: list[str] | None = None,
    data_path: Path | None = None,
) -> MatchPrediction:
    resolved_dataset = _resolve_dataset_path(data_path)
    frame = load_context_dataset(resolved_dataset)
    hypothetical = _build_hypothetical_row(frame, normalize_text(blue_team), normalize_text(red_team))

    preprocessor, feature_names, logistic_model = _load_artifacts()
    feature_frame = build_feature_frame(hypothetical)
    transformed = preprocessor.transform(feature_frame)
    base_blue_win_probability = float(logistic_model.predict_proba(transformed)[:, 1][0])

    ratings_frame = build_player_ratings()

    def parse_lineup(team_name: str, lineup_values: list[str] | None) -> dict[str, str] | None:
        if not lineup_values:
            return None
        cleaned = [normalize_text(value) for value in lineup_values if normalize_text(value)]
        if len(cleaned) != len(("top", "jng", "mid", "bot", "sup")):
            raise ValueError(f"A lineup de {team_name} precisa ter exatamente 5 jogadores: top,jng,mid,bot,sup.")
        return {position: player_name for position, player_name in zip(("top", "jng", "mid", "bot", "sup"), cleaned)}

    blue_lineup_map = parse_lineup(normalize_text(blue_team), blue_lineup)
    red_lineup_map = parse_lineup(normalize_text(red_team), red_lineup)

    blue_lineup_rating = evaluate_lineup(normalize_text(blue_team), blue_lineup_map, ratings_frame, frame)
    red_lineup_rating = evaluate_lineup(normalize_text(red_team), red_lineup_map, ratings_frame, frame)

    lineup_delta_points = blue_lineup_rating.lineup_delta - red_lineup_rating.lineup_delta
    lineup_adjustment = max(-0.12, min(0.12, lineup_delta_points * 0.0025))
    blue_win_probability = max(0.02, min(0.98, base_blue_win_probability + lineup_adjustment))
    red_win_probability = 1.0 - blue_win_probability
    series_probabilities = _best_of_five_probabilities(blue_win_probability)
    likely_blue_score, likely_red_score, likely_score_probability = _most_likely_scoreline(series_probabilities)

    game_win_probabilities = [
        {"game": 1, "blue_win_probability": blue_win_probability, "red_win_probability": 1.0 - blue_win_probability},
        {"game": 2, "blue_win_probability": blue_win_probability, "red_win_probability": 1.0 - blue_win_probability},
        {"game": 3, "blue_win_probability": blue_win_probability, "red_win_probability": 1.0 - blue_win_probability},
    ]

    common_bans = {
        "blue": _team_event_counts(frame, normalize_text(blue_team), "bans"),
        "red": _team_event_counts(frame, normalize_text(red_team), "bans"),
    }
    common_picks = {
        "blue": _team_event_counts(frame, normalize_text(blue_team), "picks"),
        "red": _team_event_counts(frame, normalize_text(red_team), "picks"),
    }

    matchup_summary = _team_history_summary(frame, normalize_text(blue_team), normalize_text(red_team))

    return MatchPrediction(
        blue_team=normalize_text(blue_team),
        red_team=normalize_text(red_team),
        base_blue_win_probability=base_blue_win_probability,
        blue_win_probability=blue_win_probability,
        red_win_probability=red_win_probability,
        model_probability=base_blue_win_probability,
        matchup_summary=matchup_summary,
        best_of=5,
        series_win_probability_blue=_best_of_five_probabilities(blue_win_probability)["blue_series_win"],
        series_win_probability_red=_best_of_five_probabilities(blue_win_probability)["red_series_win"],
        likely_series_score_blue=f"{likely_blue_score} ({likely_score_probability})",
        likely_series_score_red=f"{likely_red_score} ({likely_score_probability})",
        game_win_probabilities=game_win_probabilities,
        common_bans=common_bans,
        common_picks=common_picks,
        lineup_summary={
            "blue": {
                "lineup_rating": blue_lineup_rating.lineup_rating,
                "baseline_lineup_rating": blue_lineup_rating.baseline_lineup_rating,
                "delta": blue_lineup_rating.lineup_delta,
                "players": blue_lineup_rating.players,
                "missing_players": blue_lineup_rating.missing_players,
            },
            "red": {
                "lineup_rating": red_lineup_rating.lineup_rating,
                "baseline_lineup_rating": red_lineup_rating.baseline_lineup_rating,
                "delta": red_lineup_rating.lineup_delta,
                "players": red_lineup_rating.players,
                "missing_players": red_lineup_rating.missing_players,
            },
            "adjustment_points": lineup_delta_points,
            "adjustment_probability": lineup_adjustment,
            "base_blue_win_probability": base_blue_win_probability,
            "adjusted_blue_win_probability": blue_win_probability,
            "adjusted_red_win_probability": red_win_probability,
        },
    )


@dataclass(frozen=True)
class SeriesPrediction:
    blue_team: str
    red_team: str
    blue_rating: float
    red_rating: float
    best_of: int
    game_probabilities: list[dict[str, Any]]
    series_win_probability_blue: float
    series_win_probability_red: float
    score_probabilities: dict[str, float]
    most_likely_score: str
    side_advantage: float
    draft_weight: float


def _load_rating_setup() -> tuple[EloConfig, float]:
    if not RATING_CONFIG_FILE.exists():
        raise FileNotFoundError(
            "Configuração de rating não encontrada. Rode antes: python3 scripts/build_team_ratings.py"
        )
    payload = json.loads(RATING_CONFIG_FILE.read_text(encoding="utf-8"))
    return EloConfig(**payload["config"]), float(payload["draft_weight"])


def _draft_probability(
    frame: pd.DataFrame,
    blue_team: str,
    red_team: str,
    blue_picks: list[str],
    red_picks: list[str],
    blue_bans: list[str],
    red_bans: list[str],
) -> float:
    hypothetical = _build_hypothetical_row(frame, blue_team, red_team)
    hypothetical.loc[:, "blue_picks"] = "; ".join(blue_picks)
    hypothetical.loc[:, "red_picks"] = "; ".join(red_picks)
    hypothetical.loc[:, "blue_bans"] = "; ".join(blue_bans)
    hypothetical.loc[:, "red_bans"] = "; ".join(red_bans)
    preprocessor, _, logistic_model = _load_artifacts()
    transformed = preprocessor.transform(build_feature_frame(hypothetical))
    return float(logistic_model.predict_proba(transformed)[:, 1][0])


def predict_series(
    blue_team: str,
    red_team: str,
    best_of: int = 3,
    blue_picks: list[str] | None = None,
    red_picks: list[str] | None = None,
    blue_bans: list[str] | None = None,
    red_bans: list[str] | None = None,
    data_path: Path | None = None,
) -> SeriesPrediction:
    blue_team = normalize_text(blue_team)
    red_team = normalize_text(red_team)
    config, draft_weight = _load_rating_setup()

    frame = load_context_dataset(_resolve_dataset_path(data_path))
    known_teams = sorted(set(frame["blue_team"]) | set(frame["red_team"]))
    for team in (blue_team, red_team):
        if team not in known_teams:
            raise ValueError(f"Time desconhecido: {team}. Times disponíveis: {', '.join(known_teams)}")

    impact_lookup = build_impact_lookup()
    engine, _ = run_walk_forward(frame, config, impact_lookup)
    blue_rating = engine.rating(blue_team)
    red_rating = engine.rating(red_team)

    game1_rating_prob = expected_score(blue_rating, red_rating, config.side_advantage)
    neutral_prob = expected_score(blue_rating, red_rating, 0.0)

    used_draft = bool(blue_picks and red_picks)
    if used_draft:
        p_draft = _draft_probability(
            frame, blue_team, red_team, blue_picks or [], red_picks or [], blue_bans or [], red_bans or []
        )
        game1_prob = blend_probabilities(game1_rating_prob, p_draft, draft_weight)
    else:
        game1_prob = game1_rating_prob

    game_probs = [game1_prob] + [neutral_prob] * (best_of - 1)
    series = series_probabilities(game_probs, best_of)

    game_probabilities = [
        {
            "game": index + 1,
            "blue_win_probability": probability,
            "red_win_probability": 1.0 - probability,
            "used_draft": used_draft and index == 0,
        }
        for index, probability in enumerate(game_probs)
    ]

    return SeriesPrediction(
        blue_team=blue_team,
        red_team=red_team,
        blue_rating=round(blue_rating, 1),
        red_rating=round(red_rating, 1),
        best_of=best_of,
        game_probabilities=game_probabilities,
        series_win_probability_blue=series["a_series_win"],
        series_win_probability_red=series["b_series_win"],
        score_probabilities=series["score_probabilities"],
        most_likely_score=series["most_likely_score"],
        side_advantage=config.side_advantage,
        draft_weight=draft_weight,
    )
