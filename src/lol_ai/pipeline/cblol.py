from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Iterable

from lol_ai.config import DATA_DIR, INTERIM_DATA_DIR, LEGACY_FILTERED_FILE, PROCESSED_DATA_DIR, RAW_DATA_DIR


TEAM_POSITIONS = ("top", "jng", "mid", "bot", "sup")
FILTERED_FILE_NAME = "cblol_esports_matches_data.csv"
PROCESSED_FILE_NAME = "cblol_game_context_dataset.csv"


@dataclass(frozen=True)
class GameMeta:
    gameid: str
    series_id: str
    game_number: int
    started_at: datetime


def normalize_text(value: str | None) -> str:
    return (value or "").strip()


def is_truthy(value: str | None) -> bool:
    normalized = normalize_text(value).lower()
    return normalized not in {"", "0", "false", "no", "none"}


def should_include_row(league: str, year: str) -> bool:
    normalized_league = normalize_text(league).lower()
    normalized_year = normalize_text(year)
    if "cblol" in normalized_league and normalized_year in {"2025", "2026"}:
        return True
    return normalized_league == "lta s" and normalized_year == "2025"


def winrate(results: Iterable[bool], window: int | None = None) -> str:
    values = list(results)
    if window is not None:
        values = values[-window:]
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


def join_nonempty(values: Iterable[str]) -> str:
    cleaned = [normalize_text(value) for value in values if normalize_text(value)]
    return "; ".join(cleaned)


def get_team_row(rows: list[dict[str, str]], side: str) -> dict[str, str] | None:
    for row in rows:
        if normalize_text(row.get("position")) == "team" and normalize_text(row.get("side")) == side:
            return row
    return None


def get_player_rows(rows: list[dict[str, str]], side: str) -> dict[str, dict[str, str]]:
    players: dict[str, dict[str, str]] = {}
    for row in rows:
        if normalize_text(row.get("position")) in TEAM_POSITIONS and normalize_text(row.get("side")) == side:
            players[normalize_text(row.get("position"))] = row
    return players


def get_bans_and_picks(row: dict[str, str]) -> tuple[str, str]:
    bans = join_nonempty(row.get(f"ban{i}") for i in range(1, 6))
    picks = join_nonempty(row.get(f"pick{i}") for i in range(1, 6))
    return bans, picks


def series_score_before(current_series_winners: list[str], blue_team: str, red_team: str) -> tuple[int, int]:
    blue_wins = sum(1 for winner in current_series_winners if winner == blue_team)
    red_wins = sum(1 for winner in current_series_winners if winner == red_team)
    return blue_wins, red_wins


def matchup_winrate_before(matchup_winners: list[str], team_name: str) -> str:
    return winrate([winner == team_name for winner in matchup_winners], 10)


def role_winrate_before(
    team_position_history: DefaultDict[tuple[str, str], list[bool]],
    team_name: str,
    position: str,
) -> str:
    return winrate(team_position_history[(team_name, position)], 10)


def resolve_input_files() -> list[Path]:
    candidates = [
        RAW_DATA_DIR / "2025_LoL_esports_match_data_from_OraclesElixir.csv",
        RAW_DATA_DIR / "2026_LoL_esports_match_data_from_OraclesElixir.csv",
        DATA_DIR / "2025_LoL_esports_match_data_from_OraclesElixir.csv",
        DATA_DIR / "2026_LoL_esports_match_data_from_OraclesElixir.csv",
    ]
    return [path for path in candidates if path.exists()]


def filter_cblol_matches(
    input_files: Iterable[Path] | None = None,
    output_path: Path | None = None,
) -> Path:
    files = list(input_files) if input_files is not None else resolve_input_files()
    if not files:
        raise FileNotFoundError("Nenhum arquivo bruto de 2025/2026 foi encontrado em data/raw ou data/.")

    target_path = output_path or (INTERIM_DATA_DIR / FILTERED_FILE_NAME)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    headers: list[str] | None = None
    filtered_rows: list[dict[str, str]] = []

    for input_file in files:
        with input_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if headers is None:
                headers = reader.fieldnames or []
            for row in reader:
                if should_include_row(row.get("league", ""), row.get("year", "")):
                    filtered_rows.append(row)

    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers or [])
        writer.writeheader()
        writer.writerows(filtered_rows)

    return target_path


def load_games(input_path: Path) -> list[tuple[GameMeta, list[dict[str, str]]]]:
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            gameid = normalize_text(row.get("gameid"))
            started_at = normalize_text(row.get("date"))
            if not gameid or not started_at:
                continue
            grouped_rows[gameid].append(row)

    games: list[tuple[GameMeta, list[dict[str, str]]]] = []
    for gameid, rows in grouped_rows.items():
        blue_row = get_team_row(rows, "Blue")
        red_row = get_team_row(rows, "Red")
        if blue_row is None or red_row is None:
            continue

        blue_team = normalize_text(blue_row.get("teamname"))
        red_team = normalize_text(red_row.get("teamname"))
        if not blue_team or not red_team:
            continue

        started_at = datetime.fromisoformat(normalize_text(blue_row.get("date")))
        series_id = "|".join(
            [
                normalize_text(blue_row.get("league")),
                normalize_text(blue_row.get("year")),
                normalize_text(blue_row.get("split")),
                normalize_text(blue_row.get("playoffs")),
                started_at.date().isoformat(),
                " vs ".join(sorted((blue_team, red_team))),
            ]
        )
        game_number = int(normalize_text(blue_row.get("game")) or "0")

        games.append(
            (
                GameMeta(
                    gameid=gameid,
                    series_id=series_id,
                    game_number=game_number,
                    started_at=started_at,
                ),
                rows,
            )
        )

    games.sort(key=lambda item: (item[0].started_at, item[0].series_id, item[0].game_number, item[0].gameid))
    return games


def build_context_row(
    meta: GameMeta,
    rows: list[dict[str, str]],
    series_game_counts: dict[str, int],
    series_winners: DefaultDict[str, list[str]],
    matchup_winners: DefaultDict[tuple[str, str], list[str]],
    team_results_history: DefaultDict[str, list[bool]],
    team_position_history: DefaultDict[tuple[str, str], list[bool]],
) -> dict[str, str]:
    blue_row = get_team_row(rows, "Blue")
    red_row = get_team_row(rows, "Red")
    if blue_row is None or red_row is None:
        raise ValueError(f"Game {meta.gameid} does not contain both team rows.")

    blue_team = normalize_text(blue_row.get("teamname"))
    red_team = normalize_text(red_row.get("teamname"))
    if not blue_team or not red_team:
        raise ValueError(f"Game {meta.gameid} is missing team names.")

    blue_win = is_truthy(blue_row.get("result"))
    winner_team = blue_team if blue_win else red_team
    current_series_winners = series_winners[meta.series_id]
    blue_score_before, red_score_before = series_score_before(current_series_winners, blue_team, red_team)

    first_pick_side = "Blue" if is_truthy(blue_row.get("firstPick")) else "Red" if is_truthy(red_row.get("firstPick")) else ""
    blue_bans, blue_picks = get_bans_and_picks(blue_row)
    red_bans, red_picks = get_bans_and_picks(red_row)

    matchup_key = tuple(sorted((blue_team, red_team)))
    matchup_history = matchup_winners[matchup_key]

    blue_player_rows = get_player_rows(rows, "Blue")
    red_player_rows = get_player_rows(rows, "Red")
    blue_role_winrates = {position: role_winrate_before(team_position_history, blue_team, position) for position in TEAM_POSITIONS}
    red_role_winrates = {position: role_winrate_before(team_position_history, red_team, position) for position in TEAM_POSITIONS}

    context_text = (
        f"Series: {blue_team} vs {red_team} | Patch {normalize_text(blue_row.get('patch'))} | "
        f"Split {normalize_text(blue_row.get('split'))} | Playoffs {normalize_text(blue_row.get('playoffs'))} | "
        f"Game {meta.game_number}/{series_game_counts[meta.series_id]} | Score before: {blue_score_before}-{red_score_before} | "
        f"Blue side: {blue_team} | Red side: {red_team} | First pick: {first_pick_side} | "
        f"Blue bans: {blue_bans} | Red bans: {red_bans} | "
        f"Blue last10: {winrate(team_results_history[blue_team], 10)} | Red last10: {winrate(team_results_history[red_team], 10)} | "
        f"H2H last10: {matchup_winrate_before(matchup_history, blue_team)}-{matchup_winrate_before(matchup_history, red_team)} | "
        f"Blue roles: {', '.join(f'{pos}={blue_role_winrates[pos]}' for pos in TEAM_POSITIONS)} | "
        f"Red roles: {', '.join(f'{pos}={red_role_winrates[pos]}' for pos in TEAM_POSITIONS)}"
    )

    row: dict[str, str] = {
        "series_id": meta.series_id,
        "game_id": meta.gameid,
        "game_number": str(meta.game_number),
        "series_game_count": str(series_game_counts[meta.series_id]),
        "best_of": str(series_game_counts[meta.series_id]),
        "league": normalize_text(blue_row.get("league")),
        "year": normalize_text(blue_row.get("year")),
        "split": normalize_text(blue_row.get("split")),
        "playoffs": normalize_text(blue_row.get("playoffs")),
        "date": normalize_text(blue_row.get("date")),
        "patch": normalize_text(blue_row.get("patch")),
        "blue_team": blue_team,
        "red_team": red_team,
        "blue_win": "1" if blue_win else "0",
        "winner_team": winner_team,
        "series_games_played_before": str(len(current_series_winners)),
        "series_score_blue_before": str(blue_score_before),
        "series_score_red_before": str(red_score_before),
        "first_pick_side": first_pick_side,
        "blue_bans": blue_bans,
        "red_bans": red_bans,
        "blue_picks": blue_picks,
        "red_picks": red_picks,
        "blue_last5_winrate": winrate(team_results_history[blue_team], 5),
        "blue_last10_winrate": winrate(team_results_history[blue_team], 10),
        "red_last5_winrate": winrate(team_results_history[red_team], 5),
        "red_last10_winrate": winrate(team_results_history[red_team], 10),
        "blue_h2h_last10_winrate": matchup_winrate_before(matchup_history, blue_team),
        "red_h2h_last10_winrate": matchup_winrate_before(matchup_history, red_team),
        "context_text": context_text,
    }

    for position in TEAM_POSITIONS:
        row[f"blue_{position}_last10_winrate"] = blue_role_winrates[position]
        row[f"red_{position}_last10_winrate"] = red_role_winrates[position]

    for position in TEAM_POSITIONS:
        row[f"blue_{position}_player"] = normalize_text(blue_player_rows.get(position, {}).get("playername"))
        row[f"red_{position}_player"] = normalize_text(red_player_rows.get(position, {}).get("playername"))

    return row


def build_context_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    resolved_input = input_path or (INTERIM_DATA_DIR / FILTERED_FILE_NAME)
    if not resolved_input.exists():
        fallback = LEGACY_FILTERED_FILE if LEGACY_FILTERED_FILE.exists() else None
        if fallback is None:
            raise FileNotFoundError(
                f"Arquivo filtrado não encontrado em {resolved_input} nem no caminho legado {LEGACY_FILTERED_FILE}."
            )
        resolved_input = fallback

    games = load_games(resolved_input)
    series_game_counts = defaultdict(int)
    for meta, _ in games:
        series_game_counts[meta.series_id] += 1

    series_winners: DefaultDict[str, list[str]] = defaultdict(list)
    matchup_winners: DefaultDict[tuple[str, str], list[str]] = defaultdict(list)
    team_results_history: DefaultDict[str, list[bool]] = defaultdict(list)
    team_position_history: DefaultDict[tuple[str, str], list[bool]] = defaultdict(list)

    output_rows: list[dict[str, str]] = []
    for meta, rows in games:
        row = build_context_row(
            meta,
            rows,
            series_game_counts,
            series_winners,
            matchup_winners,
            team_results_history,
            team_position_history,
        )
        output_rows.append(row)

        blue_team = row["blue_team"]
        red_team = row["red_team"]
        blue_win = row["blue_win"] == "1"
        winner_team = blue_team if blue_win else red_team

        series_winners[meta.series_id].append(winner_team)
        matchup_key = tuple(sorted((blue_team, red_team)))
        matchup_winners[matchup_key].append(winner_team)
        team_results_history[blue_team].append(blue_win)
        team_results_history[red_team].append(not blue_win)

        for row_data in rows:
            position = normalize_text(row_data.get("position"))
            if position not in TEAM_POSITIONS:
                continue
            team_name = normalize_text(row_data.get("teamname"))
            side = normalize_text(row_data.get("side"))
            if not team_name or side not in {"Blue", "Red"}:
                continue
            team_position_history[(team_name, position)].append(blue_win if side == "Blue" else not blue_win)

    resolved_output = output_path or (PROCESSED_DATA_DIR / PROCESSED_FILE_NAME)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "series_id",
        "game_id",
        "game_number",
        "series_game_count",
        "best_of",
        "league",
        "year",
        "split",
        "playoffs",
        "date",
        "patch",
        "blue_team",
        "red_team",
        "blue_win",
        "winner_team",
        "series_games_played_before",
        "series_score_blue_before",
        "series_score_red_before",
        "first_pick_side",
        "blue_bans",
        "red_bans",
        "blue_picks",
        "red_picks",
        "blue_last5_winrate",
        "blue_last10_winrate",
        "red_last5_winrate",
        "red_last10_winrate",
        "blue_h2h_last10_winrate",
        "red_h2h_last10_winrate",
    ]

    for position in TEAM_POSITIONS:
        fieldnames.append(f"blue_{position}_last10_winrate")
    for position in TEAM_POSITIONS:
        fieldnames.append(f"red_{position}_last10_winrate")
    for position in TEAM_POSITIONS:
        fieldnames.append(f"blue_{position}_player")
    for position in TEAM_POSITIONS:
        fieldnames.append(f"red_{position}_player")

    fieldnames.append("context_text")

    with resolved_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return resolved_output
