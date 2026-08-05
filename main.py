from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lol_ai.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORT_ARTIFACTS_DIR  # noqa: E402
from lol_ai.modeling.explain import explain_model  # noqa: E402
from lol_ai.modeling.player_impact import build_player_ratings  # noqa: E402
from lol_ai.modeling.training import train_models  # noqa: E402
from lol_ai.modeling.visualization import generate_all_plots  # noqa: E402
from lol_ai.pipeline.cblol import build_context_dataset, filter_cblol_matches  # noqa: E402


def run_pipeline() -> None:
    filter_output = filter_cblol_matches(output_path=INTERIM_DATA_DIR / "cblol_esports_matches_data.csv")
    print(f"Filtro concluído: {filter_output}")

    processed_output = build_context_dataset(
        input_path=filter_output,
        output_path=PROCESSED_DATA_DIR / "cblol_game_context_dataset.csv",
    )
    print(f"Dataset contextual concluído: {processed_output}")

    training_result = train_models(processed_output)
    print("Treino concluído")
    print(training_result["metrics"]["logistic_regression"]["test"])
    print(training_result["metrics"]["xgboost"]["test"])

    explanation = explain_model(processed_output)
    print(f"SHAP concluído: {explanation['blue_win_probability']:.4f}")

    ratings = build_player_ratings()
    print(f"Medidor de jogadores gerado: {len(ratings)} jogadores")

    plot_paths = generate_all_plots()
    print("Gráficos gerados")
    for path in plot_paths:
        print(path)
    print(f"Pasta de saída dos gráficos: {REPORT_ARTIFACTS_DIR}")

    from lol_ai.modeling.rating_backtest import run_rating_backtest

    rating_payload = run_rating_backtest(processed_output)
    print("Backtest de rating concluído")
    print(rating_payload["rating"]["test"])


if __name__ == "__main__":
    run_pipeline()
