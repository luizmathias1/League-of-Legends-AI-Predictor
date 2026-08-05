from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from lol_ai.config import MODEL_ARTIFACTS_DIR, REPORT_ARTIFACTS_DIR, SHAP_ARTIFACTS_DIR
from lol_ai.modeling.features import load_context_dataset, create_dataset_split
from lol_ai.modeling.training import _to_dense


def explain_model(data_path: Path | None = None, sample_index: int = 0) -> dict[str, object]:
    frame = load_context_dataset(data_path)
    split = create_dataset_split(frame)

    with (MODEL_ARTIFACTS_DIR / "cblol_preprocessor.pkl").open("rb") as handle:
        preprocessor = pickle.load(handle)
    with (MODEL_ARTIFACTS_DIR / "cblol_xgboost.pkl").open("rb") as handle:
        xgb_model = pickle.load(handle)
    feature_names = json.loads((MODEL_ARTIFACTS_DIR / "cblol_feature_names.json").read_text(encoding="utf-8"))

    X_test = preprocessor.transform(split.X_test)
    dense_test = _to_dense(X_test)

    if dense_test.shape[0] == 0:
        raise ValueError("Não há linhas no split de teste para explicar.")

    sample_index = max(0, min(sample_index, dense_test.shape[0] - 1))
    sample = dense_test[[sample_index]]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values).reshape(-1)
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(np.asarray(base_value).reshape(-1)[0])
    else:
        base_value = float(base_value)

    sample_probability = float(xgb_model.predict_proba(sample)[:, 1][0])
    contributions = sorted(
        zip(feature_names, shap_values),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    top_positive = [
        {"feature": name, "shap_value": float(value)}
        for name, value in contributions
        if value > 0
    ][:10]
    top_negative = [
        {"feature": name, "shap_value": float(value)}
        for name, value in contributions
        if value < 0
    ][:10]

    SHAP_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = [
        {"feature": name, "mean_abs_shap": float(np.abs(shap_values[i]))}
        for i, name in enumerate(feature_names)
    ]
    pd.DataFrame(summary_rows).sort_values("mean_abs_shap", ascending=False).to_csv(
        SHAP_ARTIFACTS_DIR / "cblol_shap_sample_summary.csv",
        index=False,
    )

    report = {
        "sample_index": sample_index,
        "blue_win_probability": sample_probability,
        "base_value": base_value,
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
    }
    with (REPORT_ARTIFACTS_DIR / "cblol_shap_explanation.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    markdown = [
        "# SHAP Explanation",
        f"- Blue win probability: {sample_probability:.4f}",
        f"- Base value: {base_value:.4f}",
        "",
        "## Positive factors",
    ]
    for item in top_positive:
        markdown.append(f"- {item['feature']}: {item['shap_value']:.4f}")
    markdown.append("")
    markdown.append("## Negative factors")
    for item in top_negative:
        markdown.append(f"- {item['feature']}: {item['shap_value']:.4f}")

    (REPORT_ARTIFACTS_DIR / "cblol_shap_explanation.md").write_text("\n".join(markdown), encoding="utf-8")

    return report
