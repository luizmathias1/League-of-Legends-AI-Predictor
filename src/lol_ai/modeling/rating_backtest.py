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


def run_walk_forward_with_players(
    frame: pd.DataFrame,
    config: EloConfig,
    player_config,
    performance_lookup: dict[tuple[str, str, str], float],
    team_impact_lookup: dict[tuple[str, str], float] | None = None,
):
    """Walk-forward do time + Elo individual dos jogadores em paralelo.

    Por padrão o ajuste de roster do time usa o Elo ao vivo dos jogadores;
    passe team_impact_lookup para usar o impact score estático no time e
    manter o Elo dos jogadores apenas como métrica de qualidade."""
    from lol_ai.modeling.player_elo import LivePlayerRatingLookup, PlayerEloEngine

    player_engine = PlayerEloEngine(player_config, performance_lookup)
    roster_lookup = team_impact_lookup if team_impact_lookup is not None else LivePlayerRatingLookup(player_engine)
    team_engine = RatingEngine(config, impact_lookup=roster_lookup)
    ordered = frame.sort_values(["date", "series_id", "game_number"])
    probabilities: dict[object, float] = {}
    for index, row in ordered.iterrows():
        blue_lineup = _lineup_from_row(row, "blue")
        red_lineup = _lineup_from_row(row, "red")
        blue_win = bool(int(row["blue_win"]))
        # o ajuste de roster dentro do process_game usa o Elo dos jogadores
        # ANTES deste jogo; só depois o Elo individual é atualizado
        expected_blue = team_engine.process_game(
            date=row["date"],
            league=str(row.get("league", "")),
            year=int(row["year"]),
            blue_team=str(row["blue_team"]),
            red_team=str(row["red_team"]),
            blue_lineup=blue_lineup,
            red_lineup=red_lineup,
            blue_win=blue_win,
        )
        probabilities[index] = expected_blue
        game_id = str(row.get("game_id", ""))
        player_engine.process_side(
            date=row["date"], game_id=game_id, team=str(row["blue_team"]),
            opponent=str(row["red_team"]), lineup=blue_lineup,
            expected=expected_blue, win=blue_win,
        )
        player_engine.process_side(
            date=row["date"], game_id=game_id, team=str(row["red_team"]),
            opponent=str(row["blue_team"]), lineup=red_lineup,
            expected=1.0 - expected_blue, win=not blue_win,
        )
    return team_engine, player_engine, pd.Series(probabilities).reindex(frame.index)


PLAYER_IMPACT_SCALE_GRID = (0.0, 0.25, 0.5, 1.0)


def calibrate_config_with_players(
    frame: pd.DataFrame,
    validation_index: pd.Index,
    performance_lookup: dict[tuple[str, str, str], float],
    side_advantage: float,
    player_config,
) -> EloConfig:
    best_config: EloConfig | None = None
    best_loss = float("inf")
    y_validation = frame.loc[validation_index, "blue_win"].astype(int)
    for k, carry, roster, impact in product(K_GRID, SEASON_CARRY_GRID, ROSTER_REGRESSION_GRID, PLAYER_IMPACT_SCALE_GRID):
        config = EloConfig(
            k=k,
            season_carry=carry,
            roster_regression_per_player=roster,
            impact_scale=impact,
            side_advantage=side_advantage,
        )
        _, _, probabilities = run_walk_forward_with_players(frame, config, player_config, performance_lookup)
        loss = float(log_loss(y_validation, probabilities.loc[validation_index], labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_config = config
    assert best_config is not None
    return best_config


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
    from lol_ai.modeling.player_elo import PlayerEloConfig, build_game_performance_scores
    from lol_ai.modeling.training import evaluate_predictions

    frame = load_context_dataset(data_path)
    train_index, validation_index, test_index = chronological_series_split(frame)

    train_blue_win_rate = float(frame.loc[train_index, "blue_win"].astype(int).mean())
    side_advantage = estimate_side_advantage(train_blue_win_rate)

    player_config = PlayerEloConfig()
    performance_lookup = build_game_performance_scores()

    from lol_ai.modeling.player_impact import build_impact_lookup

    validation_start = frame.loc[validation_index, "date"].min()
    static_lookup = build_impact_lookup(cutoff_date=validation_start)
    y_validation_calib = frame.loc[validation_index, "blue_win"].astype(int)

    # Duas fontes candidatas para o ajuste de roster; a validação decide.
    best_static = calibrate_config(frame, validation_index, static_lookup, side_advantage)
    _, static_probs = run_walk_forward(frame, best_static, static_lookup)
    loss_static = float(log_loss(y_validation_calib, static_probs.loc[validation_index], labels=[0, 1]))

    best_player = calibrate_config_with_players(
        frame, validation_index, performance_lookup, side_advantage, player_config
    )
    _, _, player_probs = run_walk_forward_with_players(frame, best_player, player_config, performance_lookup)
    loss_player = float(log_loss(y_validation_calib, player_probs.loc[validation_index], labels=[0, 1]))

    if loss_player <= loss_static:
        roster_source = "player_elo"
        best_config = best_player
        final_team_lookup = None
    else:
        roster_source = "static_impact"
        best_config = best_static
        # mantém o lookup cortado na validação: as métricas de teste abaixo
        # precisam continuar sem vazamento (produção usa período completo)
        final_team_lookup = static_lookup

    engine, player_engine, probabilities = run_walk_forward_with_players(
        frame, best_config, player_config, performance_lookup, team_impact_lookup=final_team_lookup
    )

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
        "player_config": asdict(player_config),
        "roster_source": roster_source,
        "roster_validation_loss": {"static_impact": loss_static, "player_elo": loss_player},
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

    player_ranking = player_engine.ranking()
    player_ranking.to_csv(REPORT_ARTIFACTS_DIR / "player_elo_ratings.csv", index=False)
    pd.DataFrame(player_engine.history).to_csv(REPORT_ARTIFACTS_DIR / "player_elo_history.csv", index=False)

    with (REPORT_ARTIFACTS_DIR / "rating_model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    RATING_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RATING_CONFIG_FILE.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(best_config),
                "player_config": asdict(player_config),
                "roster_source": roster_source,
                "draft_weight": draft_weight,
            },
            handle, indent=2, ensure_ascii=False,
        )

    _plot_reports(payload, y_test, probabilities.loc[test_index], REPORT_ARTIFACTS_DIR, plt)
    _plot_player_ranking(player_ranking, REPORT_ARTIFACTS_DIR, plt)
    return payload


def _plot_player_ranking(player_ranking: pd.DataFrame, output_dir, plt, min_games: int = 8) -> None:
    top = player_ranking[player_ranking["games"] >= min_games].head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(top) + 1.4))
    ax.barh(range(len(top)), top["rating"] - 1500.0, left=1500.0, height=0.62, color=CATEGORICAL_COLORS[0])
    labels = [f"{row.player} ({row.position}, {row.last_team})" for row in top.itertuples()]
    ax.set_yticks(range(len(top)), labels, fontsize=8.5)
    for position, rating in enumerate(top["rating"]):
        ax.text(rating + 2, position, f"{rating:.0f}", va="center", fontsize=8, color=TEXT_COLOR)
    ax.axvline(1500.0, linestyle="--", linewidth=1, color="#999999")
    ax.set_xlabel("Elo do jogador (1500 = neutro)")
    ax.set_title(f"Top 15 jogadores por Elo individual (mín. {min_games} jogos)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "player_elo_ranking.png", dpi=150)
    plt.close(fig)


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
