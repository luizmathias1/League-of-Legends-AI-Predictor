from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from lol_ai.config import MODEL_ARTIFACTS_DIR, REPORT_ARTIFACTS_DIR
from lol_ai.modeling.features import NUMERIC_COLUMNS, build_feature_frame, create_dataset_split, load_context_dataset


@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    roc_auc: float
    log_loss: float
    f1: float
    brier: float
    confusion_matrix: list[list[int]]


def build_preprocessor(categorical_columns: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate_predictions(y_true: pd.Series, probabilities: np.ndarray) -> ModelMetrics:
    predicted = (probabilities >= 0.5).astype(int)
    return ModelMetrics(
        accuracy=float(accuracy_score(y_true, predicted)),
        precision=float(precision_score(y_true, predicted, zero_division=0)),
        recall=float(recall_score(y_true, predicted, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, probabilities)) if len(set(y_true)) > 1 else float("nan"),
        log_loss=float(log_loss(y_true, probabilities, labels=[0, 1])),
        f1=float(f1_score(y_true, predicted)),
        brier=float(brier_score_loss(y_true, probabilities)),
        confusion_matrix=confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    )


def _to_dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def train_models(data_path: Path | None = None) -> dict[str, Any]:
    frame = load_context_dataset(data_path)
    split = create_dataset_split(frame)
    feature_frame = build_feature_frame(frame)
    categorical_columns = [column for column in feature_frame.columns if column not in NUMERIC_COLUMNS]

    preprocessor = build_preprocessor(categorical_columns)
    X_train = preprocessor.fit_transform(split.X_train)
    X_validation = preprocessor.transform(split.X_validation)
    X_test = preprocessor.transform(split.X_test)

    feature_names = list(preprocessor.get_feature_names_out())

    logistic_model = LogisticRegression(max_iter=3000, class_weight="balanced")
    logistic_model.fit(X_train, split.y_train)
    logistic_validation_probabilities = logistic_model.predict_proba(X_validation)[:, 1]
    logistic_test_probabilities = logistic_model.predict_proba(X_test)[:, 1]

    xgb_model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
    )
    xgb_model.fit(
        _to_dense(X_train),
        split.y_train,
        eval_set=[(_to_dense(X_validation), split.y_validation)],
        verbose=False,
    )
    xgb_validation_probabilities = xgb_model.predict_proba(_to_dense(X_validation))[:, 1]
    xgb_test_probabilities = xgb_model.predict_proba(_to_dense(X_test))[:, 1]

    logistic_metrics = {
        "validation": asdict(evaluate_predictions(split.y_validation, logistic_validation_probabilities)),
        "test": asdict(evaluate_predictions(split.y_test, logistic_test_probabilities)),
    }
    xgb_metrics = {
        "validation": asdict(evaluate_predictions(split.y_validation, xgb_validation_probabilities)),
        "test": asdict(evaluate_predictions(split.y_test, xgb_test_probabilities)),
    }

    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with (MODEL_ARTIFACTS_DIR / "cblol_preprocessor.pkl").open("wb") as handle:
        pickle.dump(preprocessor, handle)
    with (MODEL_ARTIFACTS_DIR / "cblol_logistic_regression.pkl").open("wb") as handle:
        pickle.dump(logistic_model, handle)
    with (MODEL_ARTIFACTS_DIR / "cblol_xgboost.pkl").open("wb") as handle:
        pickle.dump(xgb_model, handle)

    metrics_payload = {
        "rows": int(frame.shape[0]),
        "train_rows": int(split.X_train.shape[0]),
        "validation_rows": int(split.X_validation.shape[0]),
        "test_rows": int(split.X_test.shape[0]),
        "logistic_regression": logistic_metrics,
        "xgboost": xgb_metrics,
    }
    with (REPORT_ARTIFACTS_DIR / "cblol_model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=False)

    predictions = pd.DataFrame(
        {
            "series_id": frame.loc[split.test_index, "series_id"].values,
            "game_id": frame.loc[split.test_index, "game_id"].values,
            "blue_team": frame.loc[split.test_index, "blue_team"].values,
            "red_team": frame.loc[split.test_index, "red_team"].values,
            "blue_win_true": split.y_test.values,
            "logistic_blue_win_proba": logistic_test_probabilities,
            "xgb_blue_win_proba": xgb_test_probabilities,
        }
    )
    predictions.to_csv(REPORT_ARTIFACTS_DIR / "cblol_test_predictions.csv", index=False)

    with (MODEL_ARTIFACTS_DIR / "cblol_feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2, ensure_ascii=False)

    return {
        "metrics": metrics_payload,
        "feature_names": feature_names,
        "xgb_model": xgb_model,
        "preprocessor": preprocessor,
        "split": split,
    }
