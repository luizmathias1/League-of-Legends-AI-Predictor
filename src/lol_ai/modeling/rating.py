from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EloConfig:
    k: float = 32.0
    initial_rating: float = 1500.0
    season_carry: float = 0.6
    roster_regression_per_player: float = 0.10
    impact_scale: float = 1.5
    side_advantage: float = 0.0


def expected_score(rating_a: float, rating_b: float, advantage_a: float = 0.0) -> float:
    return 1.0 / (1.0 + 10.0 ** (-((rating_a + advantage_a) - rating_b) / 400.0))


def elo_delta(k: float, result: float, expected: float) -> float:
    return k * (result - expected)


ROSTER_ADJUSTMENT_CAP = 60.0
NEUTRAL_IMPACT = 50.0
TEAM_POSITIONS = ("top", "jng", "mid", "bot", "sup")


class RatingEngine:
    def __init__(self, config: EloConfig, impact_lookup: dict[tuple[str, str], float] | None = None) -> None:
        self.config = config
        self.impact_lookup = impact_lookup or {}
        self.ratings: dict[str, float] = {}
        self.last_lineups: dict[str, dict[str, str]] = {}
        self.last_years: dict[str, int] = {}
        self.history: list[dict[str, object]] = []

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.config.initial_rating)

    def current_ratings(self) -> dict[str, float]:
        return dict(self.ratings)

    def _impact(self, player: str, position: str) -> float:
        return self.impact_lookup.get((player.strip().lower(), position), NEUTRAL_IMPACT)

    def _season_adjustment(self, team: str, year: int) -> float:
        last_year = self.last_years.get(team)
        if last_year is None or year <= last_year:
            return 0.0
        mean = self.config.initial_rating
        old = self.rating(team)
        new = mean + (old - mean) * self.config.season_carry
        return new - old

    def _roster_adjustment(self, team: str, lineup: dict[str, str], rating_now: float) -> tuple[int, float]:
        previous = self.last_lineups.get(team)
        if previous is None or not any(lineup.values()):
            return 0, 0.0
        changes = 0
        impact_delta = 0.0
        for position in TEAM_POSITIONS:
            new_player = (lineup.get(position) or "").strip()
            old_player = (previous.get(position) or "").strip()
            if not new_player or not old_player or new_player.lower() == old_player.lower():
                continue
            changes += 1
            impact_delta += self._impact(new_player, position) - self._impact(old_player, position)
        if changes == 0:
            return 0, 0.0
        mean = self.config.initial_rating
        keep = (1.0 - self.config.roster_regression_per_player) ** changes
        regressed = mean + (rating_now - mean) * keep
        adjustment = (regressed - rating_now) + self.config.impact_scale * impact_delta
        adjustment = max(-ROSTER_ADJUSTMENT_CAP, min(ROSTER_ADJUSTMENT_CAP, adjustment))
        return changes, adjustment

    def process_game(
        self,
        *,
        date: object,
        league: str,
        year: int,
        blue_team: str,
        red_team: str,
        blue_lineup: dict[str, str],
        red_lineup: dict[str, str],
        blue_win: bool,
    ) -> float:
        adjustments: dict[str, dict[str, float]] = {}
        for team, lineup in ((blue_team, blue_lineup), (red_team, red_lineup)):
            season_adjustment = self._season_adjustment(team, year)
            rating_after_season = self.rating(team) + season_adjustment
            roster_changes, roster_adjustment = self._roster_adjustment(team, lineup, rating_after_season)
            self.ratings[team] = rating_after_season + roster_adjustment
            adjustments[team] = {
                "season_adjustment": season_adjustment,
                "roster_changes": roster_changes,
                "roster_adjustment": roster_adjustment,
            }

        rating_blue = self.rating(blue_team)
        rating_red = self.rating(red_team)
        expected_blue = expected_score(rating_blue, rating_red, self.config.side_advantage)
        result_blue = 1.0 if blue_win else 0.0
        delta_blue = elo_delta(self.config.k, result_blue, expected_blue)

        for team, opponent, side, rating_before, expected, result, delta in (
            (blue_team, red_team, "Blue", rating_blue, expected_blue, result_blue, delta_blue),
            (red_team, blue_team, "Red", rating_red, 1.0 - expected_blue, 1.0 - result_blue, -delta_blue),
        ):
            self.ratings[team] = rating_before + delta
            self.history.append(
                {
                    "date": date,
                    "league": league,
                    "team": team,
                    "opponent": opponent,
                    "side": side,
                    "result": int(result),
                    "rating_before": round(rating_before, 2),
                    "expected": round(expected, 4),
                    "delta": round(delta, 2),
                    "rating_after": round(rating_before + delta, 2),
                    "season_adjustment": round(adjustments[team]["season_adjustment"], 2),
                    "roster_changes": int(adjustments[team]["roster_changes"]),
                    "roster_adjustment": round(adjustments[team]["roster_adjustment"], 2),
                }
            )

        self.last_lineups[blue_team] = dict(blue_lineup)
        self.last_lineups[red_team] = dict(red_lineup)
        self.last_years[blue_team] = year
        self.last_years[red_team] = year
        return expected_blue
