from __future__ import annotations

import json

import pytest

from lol_ai.config import RATING_CONFIG_FILE, REPORT_ARTIFACTS_DIR
from lol_ai.modeling.rating_backtest import run_rating_backtest


@pytest.mark.integration
def test_backtest_completo_gera_artefatos():
    payload = run_rating_backtest()

    assert (REPORT_ARTIFACTS_DIR / "team_ratings.csv").exists()
    assert (REPORT_ARTIFACTS_DIR / "team_rating_history.csv").exists()
    assert (REPORT_ARTIFACTS_DIR / "rating_model_metrics.json").exists()
    assert (REPORT_ARTIFACTS_DIR / "rating_confusion_matrix.png").exists()
    assert (REPORT_ARTIFACTS_DIR / "rating_metrics_comparison.png").exists()
    assert (REPORT_ARTIFACTS_DIR / "rating_calibration.png").exists()
    assert RATING_CONFIG_FILE.exists()

    assert 0.0 <= payload["rating"]["test"]["log_loss"]
    assert payload["rating"]["test"]["accuracy"] > 0.0
    config = json.loads(RATING_CONFIG_FILE.read_text(encoding="utf-8"))
    assert "config" in config and "draft_weight" in config
