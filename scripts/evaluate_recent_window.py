from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _bootstrap import ensure_src_on_path

PROJECT_ROOT = ensure_src_on_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lol_ai.config import RATING_CONFIG_FILE, REPORT_ARTIFACTS_DIR  # noqa: E402
from lol_ai.modeling.features import load_context_dataset  # noqa: E402
from lol_ai.modeling.player_impact import build_impact_lookup  # noqa: E402
from lol_ai.modeling.rating import EloConfig  # noqa: E402
from lol_ai.modeling.rating_backtest import run_walk_forward  # noqa: E402
from lol_ai.modeling.training import evaluate_predictions  # noqa: E402

OUTPUT_DIR = REPORT_ARTIFACTS_DIR / "recent_window"

# Paleta validada (dataviz): status good/serious com rótulo textual junto,
# categóricas azul/amarelo para comparação de duas séries.
COLOR_HIT = "#008300"
COLOR_MISS = "#e34948"
COLOR_WINDOW = "#2a78d6"
COLOR_BASELINE = "#eda100"
TEXT_COLOR = "#333333"


def evaluate_window(start: str | None, days: int) -> dict:
    setup = json.loads(RATING_CONFIG_FILE.read_text(encoding="utf-8"))
    config = EloConfig(**setup["config"])

    frame = load_context_dataset()
    if start is None:
        start_ts = frame["date"].max().normalize() - pd.Timedelta(days=days - 1)
    else:
        start_ts = pd.Timestamp(start)

    impact_lookup = build_impact_lookup(cutoff_date=start_ts)
    _, probabilities = run_walk_forward(frame, config, impact_lookup)

    games = frame.loc[frame["date"] >= start_ts].sort_values("date").copy()
    if games.empty:
        raise SystemExit(f"Nenhum jogo encontrado a partir de {start_ts.date()}.")
    games["prob_blue"] = probabilities.loc[games.index]
    games["favorito"] = games.apply(lambda g: g.blue_team if g.prob_blue >= 0.5 else g.red_team, axis=1)
    games["prob_favorito"] = games["prob_blue"].apply(lambda p: max(p, 1 - p))
    games["acertou"] = games["favorito"] == games["winner_team"]

    series_rows = []
    for _, group in games.groupby("series_id", sort=False):
        ordered = group.sort_values("game_number")
        first = ordered.iloc[0]
        predicted = first["favorito"]
        score = ordered["winner_team"].value_counts()
        actual = score.idxmax()
        series_rows.append(
            {
                "data": str(first["date"])[:10],
                "serie": f"{first['blue_team']} vs {first['red_team']}",
                "previsto": predicted,
                "vencedor": actual,
                "placar": "-".join(f"{team} {wins}" for team, wins in score.items()),
                "acertou": predicted == actual,
            }
        )
    series_table = pd.DataFrame(series_rows)

    y_true = games["blue_win"].astype(int)
    metrics = evaluate_predictions(y_true, games["prob_blue"].to_numpy())

    return {
        "start": start_ts,
        "setup": setup,
        "games": games,
        "series": series_table,
        "metrics": metrics,
    }


def plot_games_timeline(games: pd.DataFrame, output_path: Path) -> None:
    labels = [
        f"{str(g.date)[5:10]}  {g.blue_team} vs {g.red_team} (j{g.game_number})"
        for g in games.itertuples()
    ]
    colors = [COLOR_HIT if hit else COLOR_MISS for hit in games["acertou"]]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(games) + 1.6))
    positions = range(len(games))
    ax.barh(positions, games["prob_favorito"] * 100, height=0.62, color=colors)
    for position, (probability, hit, favorite) in enumerate(
        zip(games["prob_favorito"], games["acertou"], games["favorito"])
    ):
        ax.text(probability * 100 + 1, position, f"{'✓' if hit else '✗'} {favorite} {probability:.0%}",
                va="center", fontsize=8.5, color=TEXT_COLOR)
    ax.set_yticks(list(positions), labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("Confiança no favorito (%)")
    ax.axvline(50, linestyle="--", linewidth=1, color="#999999")
    ax.set_title("Previsões da janela — ✓ acerto / ✗ erro (cor e símbolo)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metrics_vs_baseline(window_metrics, baseline_test: dict | None, output_path: Path) -> None:
    import numpy as np

    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss"]
    window_values = [getattr(window_metrics, name) for name in metric_names]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(metric_names))
    all_values = list(window_values)
    if baseline_test:
        width = 0.38
        ax.bar(x - width / 2, window_values, width * 0.94, label="janela recente", color=COLOR_WINDOW)
        baseline_values = [baseline_test.get(name, float("nan")) for name in metric_names]
        all_values += [value for value in baseline_values if value == value]
        ax.bar(x + width / 2, baseline_values, width * 0.94, label="backtest completo (teste)", color=COLOR_BASELINE)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    else:
        ax.bar(x, window_values, 0.6, color=COLOR_WINDOW)
    ax.set_ylim(0, max(all_values) * 1.3)
    for position, value in zip(x, window_values):
        ax.text(position - (0.19 if baseline_test else 0), value + 0.015, f"{value:.2f}",
                ha="center", fontsize=8, color=TEXT_COLOR)
    ax.set_xticks(x, metric_names, rotation=20)
    ax.set_title("Métricas da janela vs backtest completo")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion(window_metrics, output_path: Path) -> None:
    import numpy as np

    matrix = np.array(window_metrics.confusion_matrix)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    threshold = matrix.max() / 2 if matrix.max() else 0
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > threshold else TEXT_COLOR)
    ax.set_xticks([0, 1], ["Prev. Red", "Prev. Blue"])
    ax.set_yticks([0, 1], ["Red venceu", "Blue venceu"])
    ax.set_title("Matriz de confusão da janela")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avaliar as previsões do rating numa janela recente e gerar tabelas + gráficos.")
    parser.add_argument("--start", help="Data inicial da janela (YYYY-MM-DD). Padrão: últimos --days dias do dataset.")
    parser.add_argument("--days", type=int, default=14, help="Tamanho da janela quando --start não é passado (padrão: 14)")
    args = parser.parse_args()

    result = evaluate_window(args.start, args.days)
    games, series_table, metrics, setup = result["games"], result["series"], result["metrics"], result["setup"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    games_table = games[[
        "date", "blue_team", "red_team", "game_number",
        "favorito", "prob_favorito", "winner_team", "acertou",
    ]].rename(columns={"date": "data", "winner_team": "vencedor", "game_number": "jogo"})
    games_table.to_csv(OUTPUT_DIR / "janela_jogos.csv", index=False)
    series_table.to_csv(OUTPUT_DIR / "janela_series.csv", index=False)

    hits = int(games["acertou"].sum())
    series_hits = int(series_table["acertou"].sum())
    payload = {
        "janela_inicio": str(result["start"].date()),
        "janela_fim": str(games["date"].max())[:10],
        "parametros": {**setup["config"], "draft_weight": setup["draft_weight"]},
        "jogos": {"total": int(len(games)), "acertos": hits, "erros": int(len(games)) - hits},
        "series": {"total": int(len(series_table)), "acertos": series_hits, "erros": int(len(series_table)) - series_hits},
        "metricas": asdict(metrics),
    }
    baseline_file = REPORT_ARTIFACTS_DIR / "rating_model_metrics.json"
    baseline_test = None
    if baseline_file.exists():
        baseline_test = json.loads(baseline_file.read_text(encoding="utf-8"))["rating"]["test"]
        payload["backtest_referencia"] = baseline_test
    with (OUTPUT_DIR / "janela_metricas.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    plot_games_timeline(games, OUTPUT_DIR / "janela_previsoes.png")
    plot_metrics_vs_baseline(metrics, baseline_test, OUTPUT_DIR / "janela_metricas.png")
    plot_confusion(metrics, OUTPUT_DIR / "janela_confusao.png")

    print(f"Janela: {payload['janela_inicio']} a {payload['janela_fim']}")
    print(f"Jogos: {hits}/{len(games)} acertos | Séries: {series_hits}/{len(series_table)} acertos")
    print(f"F1: {metrics.f1:.3f} | accuracy: {metrics.accuracy:.3f} | log loss: {metrics.log_loss:.3f}")
    print(f"Arquivos gerados em: {OUTPUT_DIR}")
    for name in ("janela_jogos.csv", "janela_series.csv", "janela_metricas.json",
                 "janela_previsoes.png", "janela_metricas.png", "janela_confusao.png"):
        print(f"- {name}")


if __name__ == "__main__":
    main()
