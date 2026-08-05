from __future__ import annotations

from collections import defaultdict


def _game_probability(game_probs: list[float], index: int) -> float:
    if not game_probs:
        raise ValueError("game_probs não pode ser vazio.")
    if index < len(game_probs):
        return game_probs[index]
    return game_probs[-1]


def series_probabilities(game_probs: list[float], best_of: int) -> dict[str, object]:
    if best_of not in {1, 3, 5}:
        raise ValueError(f"best_of inválido: {best_of}. Use 1, 3 ou 5.")
    wins_needed = best_of // 2 + 1
    scores: dict[str, float] = defaultdict(float)

    def walk(a_wins: int, b_wins: int, accumulated: float) -> None:
        if a_wins == wins_needed or b_wins == wins_needed:
            scores[f"{a_wins}x{b_wins}"] += accumulated
            return
        p = _game_probability(game_probs, a_wins + b_wins)
        walk(a_wins + 1, b_wins, accumulated * p)
        walk(a_wins, b_wins + 1, accumulated * (1.0 - p))

    walk(0, 0, 1.0)
    a_series_win = sum(
        probability
        for score, probability in scores.items()
        if int(score.split("x")[0]) == wins_needed
    )
    return {
        "score_probabilities": dict(scores),
        "a_series_win": a_series_win,
        "b_series_win": 1.0 - a_series_win,
        "most_likely_score": max(scores, key=scores.get),
    }
