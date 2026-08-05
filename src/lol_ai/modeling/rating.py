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
