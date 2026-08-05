from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lol_ai.modeling.prediction import predict_series  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prever série do CBLOL com rating por adversário e draft opcional.")
    parser.add_argument("blue_team", help="Time no lado azul no jogo 1")
    parser.add_argument("red_team", help="Time no lado vermelho no jogo 1")
    parser.add_argument("--best-of", type=int, choices=(1, 3, 5), default=3, help="Formato da série (padrão: Bo3)")
    parser.add_argument("--blue-picks", help="Picks do azul no jogo 1, separados por vírgula")
    parser.add_argument("--red-picks", help="Picks do vermelho no jogo 1, separados por vírgula")
    parser.add_argument("--blue-bans", help="Bans do azul no jogo 1, separados por vírgula")
    parser.add_argument("--red-bans", help="Bans do vermelho no jogo 1, separados por vírgula")
    args = parser.parse_args()

    def split_champions(value: str | None) -> list[str] | None:
        if not value:
            return None
        return [item.strip() for item in value.split(",") if item.strip()]

    prediction = predict_series(
        args.blue_team,
        args.red_team,
        best_of=args.best_of,
        blue_picks=split_champions(args.blue_picks),
        red_picks=split_champions(args.red_picks),
        blue_bans=split_champions(args.blue_bans),
        red_bans=split_champions(args.red_bans),
    )

    print(f"Confronto (Bo{prediction.best_of}): {prediction.blue_team} vs {prediction.red_team}")
    print(f"Ratings: {prediction.blue_team} {prediction.blue_rating} | {prediction.red_team} {prediction.red_rating}")
    print(f"Vantagem de lado azul: {prediction.side_advantage:+.1f} pontos | Peso do draft: {prediction.draft_weight}")
    print(f"\nChance de vencer a série: {prediction.blue_team} {prediction.series_win_probability_blue:.1%} | {prediction.red_team} {prediction.series_win_probability_red:.1%}")
    print("\nChance por jogo:")
    for game in prediction.game_probabilities:
        draft_note = " (com draft)" if game["used_draft"] else ""
        print(f"- Jogo {game['game']}{draft_note}: {prediction.blue_team} {game['blue_win_probability']:.1%} | {prediction.red_team} {game['red_win_probability']:.1%}")
    print(f"\nPlacar mais provável ({prediction.blue_team} x {prediction.red_team}): {prediction.most_likely_score}")
    print("Distribuição de placares:")
    for score, probability in sorted(prediction.score_probabilities.items(), key=lambda item: -item[1]):
        print(f"- {score}: {probability:.1%}")


if __name__ == "__main__":
    main()
