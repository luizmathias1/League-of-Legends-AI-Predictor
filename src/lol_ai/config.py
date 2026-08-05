from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
REPORT_ARTIFACTS_DIR = ARTIFACTS_DIR / "reports"
SHAP_ARTIFACTS_DIR = ARTIFACTS_DIR / "shap"

RATING_CONFIG_FILE = MODEL_ARTIFACTS_DIR / "rating_config.json"

FILTERED_DATA_FILE = INTERIM_DATA_DIR / "cblol_esports_matches_data.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "cblol_game_context_dataset.csv"
