from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from lol_ai.config import REPORT_ARTIFACTS_DIR, SHAP_ARTIFACTS_DIR


PLOT_STYLE = "whitegrid"


def _ensure_output_dir() -> Path:
    REPORT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return REPORT_ARTIFACTS_DIR


def _load_metrics() -> dict:
    metrics_path = REPORT_ARTIFACTS_DIR / "cblol_model_metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _load_predictions() -> pd.DataFrame:
    predictions_path = REPORT_ARTIFACTS_DIR / "cblol_test_predictions.csv"
    return pd.read_csv(predictions_path)


def plot_metrics_comparison() -> Path:
    sns.set_theme(style=PLOT_STYLE)
    output_dir = _ensure_output_dir()
    metrics = _load_metrics()

    rows = []
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for model_name in ("logistic_regression", "xgboost"):
        for split_name in ("validation", "test"):
            for metric_name in metric_keys:
                rows.append(
                    {
                        "model": model_name.replace("_", " ").title(),
                        "split": split_name.title(),
                        "metric": metric_name.upper() if metric_name == "roc_auc" else metric_name.title(),
                        "value": metrics[model_name][split_name][metric_name],
                    }
                )

    frame = pd.DataFrame(rows)
    g = sns.catplot(
        data=frame,
        kind="bar",
        x="metric",
        y="value",
        hue="model",
        col="split",
        height=4.2,
        aspect=1.25,
        palette="deep",
    )
    g.set_titles("{col_name} split")
    g.set_axis_labels("Métrica", "Valor")
    g._legend.set_title("Modelo")
    for ax in g.axes.flat:
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=25)
    output_path = output_dir / "metrics_comparison.png"
    g.fig.suptitle("Desempenho dos Modelos por Split", y=1.05)
    g.fig.tight_layout()
    g.fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(g.fig)
    return output_path


def plot_confusion_matrices() -> Path:
    sns.set_theme(style=PLOT_STYLE)
    output_dir = _ensure_output_dir()
    metrics = _load_metrics()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, model_name in zip(axes, ("logistic_regression", "xgboost")):
        cm = pd.DataFrame(metrics[model_name]["test"]["confusion_matrix"], index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("Predição")
        ax.set_ylabel("Real")

    output_path = output_dir / "confusion_matrices.png"
    fig.suptitle("Matriz de Confusão no Teste")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_probability_distribution() -> Path:
    sns.set_theme(style=PLOT_STYLE)
    output_dir = _ensure_output_dir()
    predictions = _load_predictions()

    long_frame = predictions.melt(
        id_vars=["blue_win_true"],
        value_vars=["logistic_blue_win_proba", "xgb_blue_win_proba"],
        var_name="model",
        value_name="probability",
    )
    long_frame["model"] = long_frame["model"].replace(
        {
            "logistic_blue_win_proba": "Logistic Regression",
            "xgb_blue_win_proba": "XGBoost",
        }
    )
    long_frame["outcome"] = long_frame["blue_win_true"].replace({0: "Blue perdeu", 1: "Blue venceu"})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, model_name in zip(axes, ["Logistic Regression", "XGBoost"]):
        subset = long_frame[long_frame["model"] == model_name]
        sns.histplot(
            data=subset,
            x="probability",
            hue="outcome",
            bins=12,
            stat="density",
            common_norm=False,
            element="step",
            ax=ax,
        )
        ax.set_title(model_name)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilidade prevista")
        ax.set_ylabel("Densidade")

    output_path = output_dir / "probability_distribution.png"
    fig.suptitle("Distribuição das Probabilidades no Teste")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_shap_importance() -> Path:
    sns.set_theme(style=PLOT_STYLE)
    output_dir = _ensure_output_dir()
    shap_path = SHAP_ARTIFACTS_DIR / "cblol_shap_sample_summary.csv"
    shap_frame = pd.read_csv(shap_path).sort_values("mean_abs_shap", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=shap_frame, y="feature", x="mean_abs_shap", ax=ax, color="#4c72b0")
    ax.set_title("Top 15 features por impacto SHAP")
    ax.set_xlabel("|SHAP| médio")
    ax.set_ylabel("Feature")
    output_path = output_dir / "shap_importance.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_plots() -> list[Path]:
    return [
        plot_metrics_comparison(),
        plot_confusion_matrices(),
        plot_probability_distribution(),
        plot_shap_importance(),
    ]
