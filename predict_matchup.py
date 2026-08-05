from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lol_ai.modeling.prediction import predict_matchup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prever uma partida do CBLOL com chance, placar provável e medidor de lineup.")
    parser.add_argument("blue_team", help="Time no lado azul")
    parser.add_argument("red_team", help="Time no lado vermelho")
    parser.add_argument(
        "--blue-lineup",
        help="Lineup do time azul no formato top,jng,mid,bot,sup",
    )
    parser.add_argument(
        "--red-lineup",
        help="Lineup do time vermelho no formato top,jng,mid,bot,sup",
    )
    args = parser.parse_args()

    blue_lineup = args.blue_lineup.split(",") if args.blue_lineup else None
    red_lineup = args.red_lineup.split(",") if args.red_lineup else None
    prediction = predict_matchup(args.blue_team, args.red_team, blue_lineup=blue_lineup, red_lineup=red_lineup)

    def print_items(title: str, items: list[dict[str, object]]) -> None:
        print(title)
        if not items:
            print("- sem dados suficientes")
            return
        for item in items:
            print(f"- {item['name']}: {item['count']} vezes ({float(item['share']):.1%})")

    print(f"Confronto: {prediction.blue_team} (Blue) vs {prediction.red_team} (Red)")
    print(f"Chance base do modelo no jogo: {prediction.blue_team} {prediction.base_blue_win_probability:.2%} | {prediction.red_team} {1.0 - prediction.base_blue_win_probability:.2%}")
    print(f"Chance ajustada pela lineup: {prediction.blue_team} {prediction.blue_win_probability:.2%} | {prediction.red_team} {prediction.red_win_probability:.2%}")
    print(f"Chance de vencer a série (Bo{prediction.best_of}): {prediction.blue_team} {prediction.series_win_probability_blue:.2%} | {prediction.red_team} {prediction.series_win_probability_red:.2%}")

    print("\nMedidor de lineup:")
    for side in ("blue", "red"):
        summary = prediction.lineup_summary[side]
        team_name = prediction.blue_team if side == "blue" else prediction.red_team
        print(f"- {team_name}: {summary['lineup_rating']:.1f}/100 (baseline {summary['baseline_lineup_rating']:.1f}/100, delta {summary['delta']:+.1f})")
        if summary["missing_players"]:
            print(f"  jogadores sem rating: {', '.join(summary['missing_players'])}")

    print("\nChance por jogo na série:")
    for game in prediction.game_win_probabilities:
        print(f"- Game {game['game']}: {prediction.blue_team} {game['blue_win_probability']:.2%} | {prediction.red_team} {game['red_win_probability']:.2%}")
    print("- Observação: sem draft e lado previstos, os jogos usam a mesma base de probabilidade.")

    print("\nPlacar final mais provável:")
    print(f"- {prediction.blue_team}: {prediction.likely_series_score_blue}")
    print(f"- {prediction.red_team}: {prediction.likely_series_score_red}")
    print(f"- ajuste por lineup: {prediction.lineup_summary['adjustment_probability']:+.2%}")

    print("\nBans comuns recentes:")
    print(f"{prediction.blue_team}")
    print_items("", prediction.common_bans["blue"])
    print(f"{prediction.red_team}")
    print_items("", prediction.common_bans["red"])

    print("\nPicks comuns recentes:")
    print(f"{prediction.blue_team}")
    print_items("", prediction.common_picks["blue"])
    print(f"{prediction.red_team}")
    print_items("", prediction.common_picks["red"])

    print("\nResumo do confronto:")
    for key, value in prediction.matchup_summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()