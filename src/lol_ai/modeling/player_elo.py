from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lol_ai.modeling.rating import TEAM_POSITIONS

# Pesos do impact score original sem a componente de resultado (renormalizados):
# o multiplicador mede desempenho ALÉM do resultado, que já entra no delta base.
PERFORMANCE_WEIGHTS = {
    "damageshare": 0.225,
    "earnedgoldshare": 0.15,
    "dpm": 0.15,
    "vspm": 0.125,
    "cspm": 0.10,
    "wardsplaced": 0.0625,
    "visionscore": 0.0625,
    "early_advantage": 0.125,
}

EARLY_COLUMNS = ("golddiffat15", "xpdiffat15", "csdiffat15")
NEUTRAL_PERFORMANCE = 50.0


@dataclass(frozen=True)
class PlayerEloConfig:
    k: float = 16.0
    initial_rating: float = 1500.0


def build_game_performance_scores(filtered_path: Path | None = None) -> dict[tuple[str, str, str], float]:
    """Percentil de desempenho (0-100) por posição, por jogo, para cada jogador.

    Chave: (game_id, nome_minusculo, posicao). Normalização global por posição —
    apenas os stats são normalizados; nenhum resultado futuro entra no score.
    """
    from lol_ai.modeling.player_impact import _resolve_filtered_path

    frame = pd.read_csv(_resolve_filtered_path(filtered_path))
    frame = frame[frame["position"].isin(TEAM_POSITIONS)].copy()
    if frame.empty:
        return {}

    for column in [*PERFORMANCE_WEIGHTS.keys() - {"early_advantage"}, *EARLY_COLUMNS]:
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    frame["early_advantage"] = frame[list(EARLY_COLUMNS)].mean(axis=1)

    score = sum(
        weight * frame.groupby("position")[column].rank(pct=True, method="average")
        for column, weight in PERFORMANCE_WEIGHTS.items()
    ) * 100.0

    return {
        (str(row_gameid), str(row_player).strip().lower(), str(row_position)): float(row_score)
        for row_gameid, row_player, row_position, row_score in zip(
            frame["gameid"], frame["playername"], frame["position"], score
        )
    }


class PlayerEloEngine:
    def __init__(
        self,
        config: PlayerEloConfig,
        performance_lookup: dict[tuple[str, str, str], float] | None = None,
    ) -> None:
        self.config = config
        self.performance_lookup = performance_lookup or {}
        self.ratings: dict[tuple[str, str], float] = {}
        self.games: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.last_team: dict[tuple[str, str], str] = {}
        self.history: list[dict[str, object]] = []

    def rating(self, player: str, position: str) -> float:
        return self.ratings.get((player.strip().lower(), position), self.config.initial_rating)

    def process_side(
        self,
        *,
        date: object,
        game_id: str,
        team: str,
        opponent: str,
        lineup: dict[str, str],
        expected: float,
        win: bool,
    ) -> None:
        base = self.config.k * ((1.0 if win else 0.0) - expected)
        for position in TEAM_POSITIONS:
            player = (lineup.get(position) or "").strip()
            if not player:
                continue
            key = (player.lower(), position)
            score = self.performance_lookup.get((game_id, player.lower(), position), NEUTRAL_PERFORMANCE)
            multiplier = 0.5 + score / 100.0 if win else 1.5 - score / 100.0
            delta = base * multiplier
            before = self.ratings.get(key, self.config.initial_rating)
            self.ratings[key] = before + delta
            self.games[key] += 1
            self.last_team[key] = team
            self.history.append(
                {
                    "date": date,
                    "game_id": game_id,
                    "player": player,
                    "position": position,
                    "team": team,
                    "opponent": opponent,
                    "result": int(win),
                    "expected": round(expected, 4),
                    "performance": round(score, 1),
                    "delta": round(delta, 2),
                    "rating_after": round(before + delta, 2),
                }
            )

    def ranking(self) -> pd.DataFrame:
        rows = [
            {
                "player": key[0],
                "position": key[1],
                "rating": round(rating, 1),
                "games": self.games[key],
                "last_team": self.last_team.get(key, ""),
            }
            for key, rating in self.ratings.items()
        ]
        return (
            pd.DataFrame(rows)
            .sort_values(["rating", "games"], ascending=[False, False])
            .reset_index(drop=True)
        )


class LivePlayerRatingLookup:
    """Adaptador para o RatingEngine: entrega o Elo atual do jogador no lugar
    do impact score estático. Jogador desconhecido vale o rating inicial."""

    def __init__(self, engine: PlayerEloEngine) -> None:
        self.engine = engine

    def get(self, key: tuple[str, str], default: object = None) -> float:
        return self.engine.ratings.get(key, self.engine.config.initial_rating)
