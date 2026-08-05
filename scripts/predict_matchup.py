from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import ensure_src_on_path


PROJECT_ROOT = ensure_src_on_path()

from lol_ai.modeling.prediction import predict_matchup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prever partida futura entre dois times do CBLOL.")
    parser.add_argument("blue_team", help="Time no lado azul, por exemplo: paiN Gaming")
    parser.add_argument("red_team", help="Time no lado vermelho, por exemplo: Vivo Keyd Stars")
    args = parser.parse_args()

    prediction = predict_matchup(args.blue_team, args.red_team)
    print(f"Confronto: {prediction.blue_team} (Blue) vs {prediction.red_team} (Red)")
    print(f"Probabilidade de vitória do Blue: {prediction.blue_win_probability:.2%}")
    print(f"Probabilidade de vitória do Red: {prediction.red_win_probability:.2%}")
    print("Resumo do confronto:")
    for key, value in prediction.matchup_summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()