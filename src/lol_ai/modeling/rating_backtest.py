from __future__ import annotations

import math
from itertools import product

import pandas as pd
from sklearn.metrics import log_loss

from lol_ai.modeling.rating import TEAM_POSITIONS, EloConfig, RatingEngine

K_GRID = (16.0, 24.0, 32.0, 40.0)
SEASON_CARRY_GRID = (0.4, 0.6, 0.8, 1.0)
ROSTER_REGRESSION_GRID = (0.0, 0.05, 0.10, 0.20)
IMPACT_SCALE_GRID = (0.0, 1.0, 2.0)


def estimate_side_advantage(blue_win_rate: float) -> float:
    clamped = min(max(blue_win_rate, 0.05), 0.95)
    return -400.0 * math.log10(1.0 / clamped - 1.0)


def _lineup_from_row(row: pd.Series, prefix: str) -> dict[str, str]:
    return {
        position: str(row.get(f"{prefix}_{position}_player") or "").strip()
        for position in TEAM_POSITIONS
    }


def run_walk_forward(
    frame: pd.DataFrame,
    config: EloConfig,
    impact_lookup: dict[tuple[str, str], float],
) -> tuple[RatingEngine, pd.Series]:
    engine = RatingEngine(config, impact_lookup)
    ordered = frame.sort_values(["date", "series_id", "game_number"])
    probabilities: dict[object, float] = {}
    for index, row in ordered.iterrows():
        probabilities[index] = engine.process_game(
            date=row["date"],
            league=str(row.get("league", "")),
            year=int(row["year"]),
            blue_team=str(row["blue_team"]),
            red_team=str(row["red_team"]),
            blue_lineup=_lineup_from_row(row, "blue"),
            red_lineup=_lineup_from_row(row, "red"),
            blue_win=bool(int(row["blue_win"])),
        )
    return engine, pd.Series(probabilities).reindex(frame.index)


def calibrate_config(
    frame: pd.DataFrame,
    validation_index: pd.Index,
    impact_lookup: dict[tuple[str, str], float],
    side_advantage: float,
) -> EloConfig:
    best_config: EloConfig | None = None
    best_loss = float("inf")
    y_validation = frame.loc[validation_index, "blue_win"].astype(int)
    for k, carry, roster, impact in product(K_GRID, SEASON_CARRY_GRID, ROSTER_REGRESSION_GRID, IMPACT_SCALE_GRID):
        config = EloConfig(
            k=k,
            season_carry=carry,
            roster_regression_per_player=roster,
            impact_scale=impact,
            side_advantage=side_advantage,
        )
        _, probabilities = run_walk_forward(frame, config, impact_lookup)
        loss = float(log_loss(y_validation, probabilities.loc[validation_index], labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_config = config
    assert best_config is not None
    return best_config


def _logit(probability: float) -> float:
    clamped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clamped / (1.0 - clamped))


def blend_probabilities(p_rating: float, p_draft: float, weight: float) -> float:
    combined = _logit(p_rating) + weight * (_logit(p_draft) - _logit(0.5))
    return 1.0 / (1.0 + math.exp(-combined))


def fit_draft_weight(p_rating: pd.Series, p_draft: pd.Series, y_true: pd.Series) -> float:
    best_weight = 0.0
    best_loss = float("inf")
    for step in range(11):
        weight = step / 10.0
        blended = [
            blend_probabilities(rating_prob, draft_prob, weight)
            for rating_prob, draft_prob in zip(p_rating, p_draft)
        ]
        loss = float(log_loss(y_true, blended, labels=[0, 1]))
        if loss < best_loss - 1e-9:
            best_loss = loss
            best_weight = weight
    return best_weight


def draft_model_probabilities(frame: pd.DataFrame, index: pd.Index) -> pd.Series:
    import json
    import pickle

    from lol_ai.config import MODEL_ARTIFACTS_DIR
    from lol_ai.modeling.features import build_feature_frame

    with (MODEL_ARTIFACTS_DIR / "cblol_preprocessor.pkl").open("rb") as handle:
        preprocessor = pickle.load(handle)
    with (MODEL_ARTIFACTS_DIR / "cblol_logistic_regression.pkl").open("rb") as handle:
        model = pickle.load(handle)
    features = build_feature_frame(frame.loc[index])
    transformed = preprocessor.transform(features)
    return pd.Series(model.predict_proba(transformed)[:, 1], index=index)


# Quatro primeiros slots da paleta categórica validada (dataviz, modo claro)
CATEGORICAL_COLORS = ("#2a78d6", "#008300", "#e87ba4", "#eda100")
TEXT_COLOR = "#333333"


def run_rating_backtest(data_path=None) -> dict:
    import json
    from dataclasses import asdict

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lol_ai.config import RATING_CONFIG_FILE, REPORT_ARTIFACTS_DIR
    from lol_ai.modeling.features import chronological_series_split, load_context_dataset
    from lol_ai.modeling.player_impact import build_impact_lookup
    from lol_ai.modeling.training import evaluate_predictions

    frame = load_context_dataset(data_path)
    train_index, validation_index, test_index = chronological_series_split(frame)

    train_blue_win_rate = float(frame.loc[train_index, "blue_win"].astype(int).mean())
    side_advantage = estimate_side_advantage(train_blue_win_rate)

    validation_start = frame.loc[validation_index, "date"].min()
    impact_lookup = build_impact_lookup(cutoff_date=validation_start)

    best_config = calibrate_config(frame, validation_index, impact_lookup, side_advantage)
    engine, probabilities = run_walk_forward(frame, best_config, impact_lookup)

    y_validation = frame.loc[validation_index, "blue_win"].astype(int)
    y_test = frame.loc[test_index, "blue_win"].astype(int)
    metrics_asdict = asdict

    p_draft_validation = draft_model_probabilities(frame, validation_index)
    p_draft_test = draft_model_probabilities(frame, test_index)
    draft_weight = fit_draft_weight(probabilities.loc[validation_index], p_draft_validation, y_validation)
    blended_test = pd.Series(
        [
            blend_probabilities(rating_prob, draft_prob, draft_weight)
            for rating_prob, draft_prob in zip(probabilities.loc[test_index], p_draft_test)
        ],
        index=test_index,
    )

    payload = {
        "rows": int(len(frame)),
        "train_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "test_rows": int(len(test_index)),
        "side_advantage": side_advantage,
        "train_blue_win_rate": train_blue_win_rate,
        "config": asdict(best_config),
        "draft_weight": draft_weight,
        "rating": {
            "validation": metrics_asdict(evaluate_predictions(y_validation, probabilities.loc[validation_index].to_numpy())),
            "test": metrics_asdict(evaluate_predictions(y_test, probabilities.loc[test_index].to_numpy())),
            "test_series_accuracy": series_level_accuracy(frame, probabilities, test_index),
        },
        "rating_plus_draft": {
            "test": metrics_asdict(evaluate_predictions(y_test, blended_test.to_numpy())),
        },
    }

    baseline_file = REPORT_ARTIFACTS_DIR / "cblol_model_metrics.json"
    if baseline_file.exists():
        payload["baselines"] = json.loads(baseline_file.read_text(encoding="utf-8"))

    REPORT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    ratings_frame = (
        pd.DataFrame(
            [{"team": team, "rating": round(rating, 1)} for team, rating in engine.current_ratings().items()]
        )
        .sort_values("rating", ascending=False)
        .reset_index(drop=True)
    )
    ratings_frame.to_csv(REPORT_ARTIFACTS_DIR / "team_ratings.csv", index=False)
    pd.DataFrame(engine.history).to_csv(REPORT_ARTIFACTS_DIR / "team_rating_history.csv", index=False)

    with (REPORT_ARTIFACTS_DIR / "rating_model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    RATING_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RATING_CONFIG_FILE.open("w", encoding="utf-8") as handle:
        json.dump({"config": asdict(best_config), "draft_weight": draft_weight}, handle, indent=2, ensure_ascii=False)

    _plot_reports(payload, y_test, probabilities.loc[test_index], REPORT_ARTIFACTS_DIR, plt)
    return payload


def _plot_reports(payload, y_test, p_test, output_dir, plt) -> None:
    import numpy as np

    # 1. Matriz de confusão do rating no teste (sequencial de um matiz)
    matrix = np.array(payload["rating"]["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    threshold = matrix.max() / 2 if matrix.max() else 0
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > threshold else TEXT_COLOR)
    ax.set_xticks([0, 1], ["Prev. Red", "Prev. Blue"])
    ax.set_yticks([0, 1], ["Red venceu", "Blue venceu"])
    ax.set_title("Rating — matriz de confusão (teste)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "rating_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 2. Comparação de métricas: rating vs rating+draft vs baselines
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "brier", "log_loss"]
    systems = {"rating": payload["rating"]["test"], "rating+draft": payload["rating_plus_draft"]["test"]}
    baselines = payload.get("baselines", {})
    for name in ("logistic_regression", "xgboost"):
        if name in baselines:
            systems[name] = baselines[name]["test"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metric_names))
    width = 0.8 / len(systems)
    for offset, (label, metrics) in enumerate(systems.items()):
        values = [metrics.get(metric, float("nan")) for metric in metric_names]
        ax.bar(x + offset * width, values, width * 0.92,
               label=label, color=CATEGORICAL_COLORS[offset % len(CATEGORICAL_COLORS)])
    ax.set_xticks(x + width * (len(systems) - 1) / 2, metric_names, rotation=20)
    ax.set_title("Métricas no teste — rating vs modelos anteriores")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "rating_metrics_comparison.png", dpi=150)
    plt.close(fig)

    # 3. Calibração + distribuição de probabilidades
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = np.linspace(0.0, 1.0, 6)
    centers = (bins[:-1] + bins[1:]) / 2
    observed = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (p_test >= low) & (p_test < high)
        observed.append(float(y_test[mask].mean()) if mask.any() else float("nan"))
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#999999", label="calibração perfeita")
    axes[0].plot(centers, observed, marker="o", markersize=8, linewidth=2,
                 color=CATEGORICAL_COLORS[0], label="rating")
    axes[0].set_title("Calibração (teste)")
    axes[0].set_xlabel("Probabilidade prevista")
    axes[0].set_ylabel("Frequência observada")
    axes[0].legend(frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].hist(p_test, bins=20, color=CATEGORICAL_COLORS[0])
    axes[1].set_title("Distribuição das probabilidades")
    axes[1].set_xlabel("Probabilidade prevista (azul)")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "rating_calibration.png", dpi=150)
    plt.close(fig)


def series_level_accuracy(
    frame: pd.DataFrame,
    probabilities: pd.Series,
    evaluation_index: pd.Index,
) -> float | None:
    rows = frame.loc[evaluation_index].copy()
    rows["blue_prob"] = probabilities.loc[evaluation_index]
    hits: list[bool] = []
    for _, group in rows.groupby("series_id"):
        ordered = group.sort_values("game_number")
        first = ordered.iloc[0]
        predicted = first["blue_team"] if first["blue_prob"] >= 0.5 else first["red_team"]
        actual = ordered["winner_team"].value_counts().idxmax()
        hits.append(predicted == actual)
    if not hits:
        return None
    return float(sum(hits) / len(hits))
